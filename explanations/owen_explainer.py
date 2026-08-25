"""Owen-value explainers for reference point-based decision support.

The explainers in this module learn a surrogate mapping from reference points
to objective vectors and use SHAP's partition explainer to estimate Owen values.
The multi-target variant aggregates a selected set of objective values into a
single achievement-scalarizing utility before computing explanations.

In addition to Owen values, the module can evaluate direct input-coalition
contributions and pairwise interactions. Owen values and direct coalition
contributions describe different effects and therefore need not have matching
signs.

The resulting Owen values are conditional on the feature hierarchy used by
SHAP's partition masker and on the empirical masking distribution represented
by the selected background data. They also depend on the fitted surrogate, the
selected target coalition, target weights, normalization, and the target utility
definition.

Owen values and direct coalition contributions describe different effects and
therefore need not have matching signs.
"""

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import polars as pl
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_default_surrogate(random_state: int = 1, surrogate_type: str = "random_forest"):
    """Construct a default multi-output surrogate model.

    Args:
        random_state (int, optional): random state used by the estimator. Defaults
        to 1.
        surrogate_type (str, optional): either ``"random_forest"`` or
        ``"gaussian_process"``. Defaults to ``"random_forest"``.

    Raises:
        ValueError: the requested surrogate type is not supported.

    Returns:
        MultiOutputRegressor: an unfitted multi-output regression model.
    """
    if surrogate_type == "random_forest":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
        )

    if surrogate_type == "gaussian_process":
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(1.0, (1e-3, 1e3))
            + WhiteKernel(1e-5, (1e-8, 1e-1))
        )
        base_model = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-8,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=random_state,
            ),
        )
        return MultiOutputRegressor(base_model)

    raise ValueError(
        f"Unknown surrogate_type: {surrogate_type}. "
        "Use 'random_forest' or 'gaussian_process'."
    )


class OwenExplainer:
    """Explain a surrogate mapping from reference points to objective vectors.

    The class fits or receives a multi-output surrogate and configures SHAP's
    partition explainer over the reference point components.
    """
    def __init__(
        self,
        problem_data: pl.DataFrame,
        input_symbols: list[str],
        output_symbols: list[str],
        surrogate_model=None,
        random_state: int = 1,
        fit_surrogate: bool = True,
    ):
        """Initialize an Owen explainer.

        Args:
            problem_data (pl.DataFrame): reference point--solution training data.
            input_symbols (list[str]): surrogate input columns.
            output_symbols (list[str]): surrogate output columns.
            surrogate_model: optional multi-output surrogate. A default random forest
            is created when omitted.
            random_state (int, optional): random state for the default surrogate.
            Defaults to 1.
            fit_surrogate (bool, optional): whether the surrogate is fitted during
            initialization. Defaults to True.
        """
        self.data = problem_data
        self.input_symbols = input_symbols
        self.output_symbols = output_symbols
        self.input_array = self.data[self.input_symbols].to_numpy()
        self.output_array = self.data[self.output_symbols].to_numpy()
        self.surrogate_model = (
            make_default_surrogate(random_state=random_state)
            if surrogate_model is None
            else surrogate_model
        )
        if fit_surrogate:
            self.surrogate_model.fit(self.input_array, self.output_array)
        self.explainer = None
        self.background_array: np.ndarray | None = None

    def evaluate(self, evaluate_array: np.ndarray) -> np.ndarray:
        """Predict objective vectors for one or more reference points.

        Args:
            evaluate_array (np.ndarray): reference point array accepted by the fitted
            surrogate.

        Returns:
            np.ndarray: predicted objective vectors.
        """
        return self.surrogate_model.predict(np.atleast_2d(evaluate_array))

    def setup(
        self,
        background_data: pl.DataFrame,
        clustering: str | np.ndarray = "correlation",
    ) -> None:
        """Configure the partition masker and SHAP explainer.

        Args:
            background_data (pl.DataFrame): background reference points used for
            feature masking.
            clustering (str | np.ndarray, optional): partition clustering accepted by
            ``shap.maskers.Partition``. Defaults to ``"correlation"``.
        """
        self.background_array = (
            background_data[self.input_symbols].to_numpy().astype(float, copy=True)
        )
        masker = shap.maskers.Partition(
            self.background_array,
            clustering=clustering,
        )
        self.explainer = shap.PartitionExplainer(
            self.evaluate,
            masker=masker,
            feature_names=self.input_symbols,
        )

    def explain_input(self, to_be_explained: pl.DataFrame):
        """Compute Owen-value explanations for reference points.

        Args:
            to_be_explained (pl.DataFrame): reference points containing all configured
            input columns.

        Raises:
            RuntimeError: ``setup`` has not been called.

        Returns:
            shap.Explanation: explanation returned by SHAP's partition explainer.
        """
        if self.explainer is None:
            raise RuntimeError("Call setup(...) before explain_input(...).")
        return self.explainer(
            to_be_explained[self.input_symbols].to_numpy()
        )


class MultiTargetOwenExplainer(OwenExplainer):
    """Explain the utility of a selected coalition of target objectives.

    The selected surrogate outputs are optionally normalized and aggregated with
    an achievement-scalarizing function. Partition SHAP then explains this scalar
    coalition utility in terms of reference point components.
    """
    def __init__(
        self,
        problem_data: pl.DataFrame,
        input_symbols: list[str],
        output_symbols: list[str],
        target_symbols: list[str],
        weights: Sequence[float] | None = None,
        minimize: bool = True,
        normalize_targets: bool = True,
        asf_rho: float = 1e-6,
        surrogate_model=None,
        random_state: int = 1,
        fit_surrogate: bool = True,
    ):
        """Initialize a multi-target Owen explainer.

        Args:
            problem_data (pl.DataFrame): reference point--solution training data.
            input_symbols (list[str]): surrogate input columns.
            output_symbols (list[str]): surrogate output columns.
            target_symbols (list[str]): selected surrogate outputs forming the target
            coalition.
            weights (Sequence[float] | None, optional): target weights. Equal weights
            are used when omitted. Defaults to None.
            minimize (bool, optional): whether surrogate outputs use minimization
            orientation. Defaults to True.
            normalize_targets (bool, optional): whether target outputs are normalized
            using their observed ranges. Defaults to True.
            asf_rho (float, optional): augmentation coefficient in the target utility.
            Defaults to 1e-6.
            surrogate_model: optional fitted or unfitted multi-output surrogate.
            random_state (int, optional): random state for a default surrogate.
            Defaults to 1.
            fit_surrogate (bool, optional): whether the surrogate is fitted during
            initialization. Defaults to True.

        Raises:
            ValueError: no targets are supplied, an unknown target is supplied, or the
            target weights are invalid.
        """
        super().__init__(
            problem_data=problem_data,
            input_symbols=input_symbols,
            output_symbols=output_symbols,
            surrogate_model=surrogate_model,
            random_state=random_state,
            fit_surrogate=fit_surrogate,
        )

        if not target_symbols:
            raise ValueError("target_symbols must contain at least one target.")

        duplicate_targets = list(
            dict.fromkeys(
                target
                for target in target_symbols
                if target_symbols.count(target) > 1
            )
        )

        if duplicate_targets:
            raise ValueError(
                "Each target symbol may be selected only once. "
                f"Duplicate target symbols: {duplicate_targets}."
            )

        unknown_targets = [
            target
            for target in target_symbols
            if target not in output_symbols
        ]

        if unknown_targets:
            raise ValueError(
                f"Unknown target symbols: {unknown_targets}. "
                f"Available outputs: {output_symbols}"
            )

        self.target_symbols = list(target_symbols)
        self.target_indices = [
            output_symbols.index(target) for target in target_symbols
        ]
        self.minimize = minimize
        self.normalize_targets = normalize_targets
        self.asf_rho = asf_rho

        if weights is None:
            validated_weights = np.ones(
                len(self.target_symbols),
                dtype=float,
            )
        else:
            validated_weights = np.asarray(
                weights,
                dtype=float,
            ).reshape(-1)

        if validated_weights.size != len(self.target_symbols):
            raise ValueError(
                "The number of weights must match the number of selected "
                "target objectives. "
                f"Received {validated_weights.size} weights for "
                f"{len(self.target_symbols)} targets."
            )

        if not np.all(np.isfinite(validated_weights)):
            raise ValueError(
                "Target weights must contain only finite values."
            )

        if np.any(validated_weights < 0.0):
            raise ValueError(
                "Target weights must be nonnegative."
            )

        weight_sum = float(validated_weights.sum())

        if np.isclose(weight_sum, 0.0):
            raise ValueError(
                "At least one target weight must be positive."
            )

        self.weights = validated_weights / weight_sum

        target_outputs = self.output_array[:, self.target_indices]
        self.target_mins = np.min(target_outputs, axis=0)
        self.target_maxs = np.max(target_outputs, axis=0)
        self.target_ranges = self.target_maxs - self.target_mins
        self.target_ranges = np.where(
            np.isclose(self.target_ranges, 0.0),
            1.0,
            self.target_ranges,
        )

    def evaluate_target_coalition(self, evaluate_array: np.ndarray) -> np.ndarray:
        """Evaluate the achievement-scalarizing utility of the target coalition.

        Args:
            evaluate_array (np.ndarray): reference points to evaluate.

        Returns:
            np.ndarray: one scalar target-coalition utility for each input row. Larger
            values represent better utility when ``minimize`` is true because the ASF
            value is negated.
        """
        outputs = self.evaluate(evaluate_array)
        targets = outputs[:, self.target_indices]

        if self.normalize_targets:
            targets = (targets - self.target_mins) / self.target_ranges

        weighted_targets = targets * self.weights
        asf_value = (
            np.max(weighted_targets, axis=1)
            + self.asf_rho * np.sum(weighted_targets, axis=1)
        )
        return np.asarray(
            -asf_value if self.minimize else asf_value,
            dtype=float,
        ).reshape(-1)

    def setup(
        self,
        background_data: pl.DataFrame,
        clustering: str | np.ndarray = "correlation",
    ) -> None:
        """Configure the partition masker and SHAP explainer.

        Args:
            background_data (pl.DataFrame): background reference points used for
            feature masking.
            clustering (str | np.ndarray, optional): partition clustering accepted by
            ``shap.maskers.Partition``. Defaults to ``"correlation"``.
        """
        self.background_array = (
            background_data[self.input_symbols].to_numpy().astype(float, copy=True)
        )
        masker = shap.maskers.Partition(
            self.background_array,
            clustering=clustering,
        )
        self.explainer = shap.PartitionExplainer(
            self.evaluate_target_coalition,
            masker=masker,
            feature_names=self.input_symbols,
        )

    def evaluate_reference_point(self, point: np.ndarray) -> np.ndarray:
        """Evaluate target-coalition utility at a reference point.

        Args:
            point (np.ndarray): one reference point in surrogate input orientation.

        Returns:
            np.ndarray: target-coalition utility as a one-dimensional array.
        """
        return self.evaluate_target_coalition(np.atleast_2d(point))

    def evaluate_input_coalition(
        self,
        point: np.ndarray,
        coalition_indices: Sequence[int],
    ) -> float:
        """Evaluate an input coalition against the background distribution.

        Components included in the coalition are fixed to the explained reference
        point. Components outside it retain their values from each background row.
        The returned value is the mean target-coalition utility over these masked rows.

        Args:
            point (np.ndarray): reference point being explained.
            coalition_indices (Sequence[int]): indices of input components included in
            the coalition.

        Raises:
            RuntimeError: ``setup`` has not been called.
            ValueError: the reference point dimension does not match the configured
            input symbols.

        Returns:
            float: expected target-coalition utility for the input coalition.
        """
        if self.background_array is None:
            raise RuntimeError("Call setup(...) before coalition evaluation.")

        point_1d = np.asarray(point, dtype=float).reshape(-1)
        if point_1d.size != len(self.input_symbols):
            raise ValueError(
                f"Expected {len(self.input_symbols)} input values, "
                f"got {point_1d.size}."
            )

        indices = list(coalition_indices)
        masked = self.background_array.copy()
        if indices:
            masked[:, indices] = point_1d[indices]

        return float(np.mean(self.evaluate_target_coalition(masked)))


    def evaluate_all_coalitions(
        self,
        point: np.ndarray,
    ) -> pl.DataFrame:
        """Evaluate every non-empty coalition of reference point components.

        Each contribution is measured relative to the empty-coalition value. These
        direct coalition contributions are not Owen values, and their signs need
        not match the corresponding average marginal allocations.

        For ``m`` input components, the method evaluates ``2**m - 1`` non-empty
        coalitions. The computational cost therefore grows exponentially with the
        number of objectives. The current implementation is intended primarily for
        low-dimensional interactive multiobjective optimization problems, such as
        the four- and five-objective examples used by this prototype.

        Args:
            point (np.ndarray): reference point being explained.

        Returns:
            pl.DataFrame: coalition values, baseline-relative contributions,
            absolute contributions, and qualitative coalition-level roles.
        """
        empty_value = self.evaluate_input_coalition(point, ())
        rows: list[dict] = []

        for size in range(1, len(self.input_symbols) + 1):
            for indices in combinations(range(len(self.input_symbols)), size):
                symbols = [self.input_symbols[index] for index in indices]
                coalition_value = self.evaluate_input_coalition(point, indices)
                # This is a direct baseline-relative coalition effect,
                # not an Owen value averaged over coalition orderings.
                contribution = coalition_value - empty_value

                rows.append(
                    {
                        "coalition": symbols,
                        "coalition_text": "{" + ", ".join(symbols) + "}",
                        "coalition_size": size,
                        "coalition_value": coalition_value,
                        "contribution": contribution,
                        "absolute_contribution": abs(contribution),
                        "coalition_level_role": (
                            "increases target utility relative to baseline"
                            if contribution > 0
                            else "decreases target utility relative to baseline"
                            if contribution < 0
                            else "no change relative to baseline"
                        ),
                    }
                )

        return pl.DataFrame(rows).sort(
            "absolute_contribution",
            descending=True,
        )

    def evaluate_pairwise_interactions(
        self,
        point: np.ndarray,
        coalition_table: pl.DataFrame | None = None,
        tolerance: float = 1e-12,
    ) -> pl.DataFrame:
        """Estimate optional pairwise interaction diagnostics.

        This method is not used by the default interactive explanation workflow.
        It is provided for experiments and detailed analysis of how pairs of
        reference-point components jointly affect the target-coalition utility.

        The interaction for a pair is its direct coalition contribution minus the
        sum of the two singleton contributions.

        Args:
            point (np.ndarray): reference point being explained.
            coalition_table (pl.DataFrame | None, optional): previously evaluated
                coalition table. If omitted, all coalitions are evaluated.
                Defaults to None.
            tolerance (float, optional): absolute threshold used to classify an
                interaction as approximately additive. Defaults to 1e-12.

        Returns:
            pl.DataFrame: pair contributions and positive, negative, or
            approximately additive interaction diagnostics.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative.")

        table = (
            self.evaluate_all_coalitions(point)
            if coalition_table is None
            else coalition_table
        )

        contributions = {
            frozenset(row["coalition"]): float(row["contribution"])
            for row in table.to_dicts()
        }

        rows: list[dict] = []

        for first_index, second_index in combinations(
            range(len(self.input_symbols)),
            2,
        ):
            first = self.input_symbols[first_index]
            second = self.input_symbols[second_index]

            pair = contributions[frozenset({first, second})]

            interaction = (
                pair
                - contributions[frozenset({first})]
                - contributions[frozenset({second})]
            )

            rows.append(
                {
                    "coalition": f"{{{first}, {second}}}",
                    "pair_contribution": pair,
                    "interaction": interaction,
                    "absolute_interaction": abs(interaction),
                    "interaction_type": (
                        "positive interaction"
                        if interaction > tolerance
                        else "negative interaction"
                        if interaction < -tolerance
                        else "approximately additive"
                    ),
                }
            )

        return pl.DataFrame(rows).sort(
            "absolute_interaction",
            descending=True,
        )


def summarize_owen_values(explanation, input_symbols: list[str]) -> pl.DataFrame:
    """Convert a SHAP explanation into a sorted Owen-value table.

    Args:
        explanation: a SHAP explanation containing values for one explained
        reference point.
        input_symbols (list[str]): symbols corresponding to the explained input
        dimensions.

    Raises:
        ValueError: the explanation cannot be reduced to one value per input
        symbol.

    Returns:
        pl.DataFrame: Owen values sorted by absolute magnitude, together with their
        average marginal roles.
    """
    values = np.asarray(explanation.values)

    if values.ndim == 2:
        values = values[0]
    elif values.ndim == 3 and values.shape[2] == 1:
        values = values[0, :, 0]
    elif values.ndim == 3 and values.shape[1] == 1:
        values = values[0, 0, :]
    elif values.ndim == 3 and values.shape[1] == len(input_symbols):
        values = values[0].mean(axis=1)
    else:
        values = np.squeeze(values)

    values = np.asarray(values, dtype=float).reshape(-1)

    if values.shape[0] != len(input_symbols):
        raise ValueError(
            f"Expected {len(input_symbols)} Owen values, got shape "
            f"{values.shape}. Original explanation.values shape was "
            f"{np.asarray(explanation.values).shape}."
        )

    return pl.DataFrame(
        {
            "input": input_symbols,
            "owen_value": values,
            "absolute_owen_value": np.abs(values),
            "average_marginal_role": np.where(
                values > 0,
                "positive average marginal effect",
                np.where(
                    values < 0,
                    "negative average marginal effect",
                    "neutral average marginal effect",
                ),
            ),
        }
    ).sort("absolute_owen_value", descending=True)
