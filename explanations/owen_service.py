"""Reusable multi-target Owen explanation service."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from explanations.owen_explainer import (
    make_default_surrogate,
)
from explanations.rximo_cases import clean_symbol
from explanations.utils import (
    get_rximo_symbols,
    validate_background_data,
)

from explanations.owen_runtime import (
    generate_explanation_result,
)


class OwenExplanationService:
    """Prepare and generate multi-target Owen explanations.

    The surrogate is trained once when the service is initialized. Individual
    target-specific Owen explainers reuse the fitted surrogate.
    """

    def __init__(
        self,
        *,
        problem,
        background_data_path: str | Path,
        surrogate_type: str = "gaussian_process",
        seed: int = 1,
        background_size: int = 100,
        coalition_advantage_threshold: float = 0.05,
    ):
        self.problem = problem

        self.background_data_path = Path(background_data_path)
        self.surrogate_type = surrogate_type
        self.seed = seed
        self.background_size = background_size

        self.coalition_advantage_threshold = (
            coalition_advantage_threshold
        )

        self.input_symbols, self.output_symbols = get_rximo_symbols(
            problem
        )

        self.data = self._load_background_data()

        self.surrogate = self._train_surrogate()

    def _load_background_data(self) -> pl.DataFrame:
        """Load and validate the RPM background dataset."""
        if not self.background_data_path.exists():
            raise FileNotFoundError(
                "Owen background dataset was not found at "
                f"{self.background_data_path}."
            )

        data = pl.read_csv(self.background_data_path)

        validate_background_data(
            data,
            input_symbols=self.input_symbols,
            output_symbols=self.output_symbols,
            source=str(self.background_data_path),
        )

        return data

    def _train_surrogate(self):
        """Train the shared reference-point-to-solution surrogate."""
        print(
            f"Training Owen surrogate ({self.surrogate_type}) "
            f"using {self.data.height} RPM samples..."
        )

        surrogate = make_default_surrogate(
            random_state=self.seed,
            surrogate_type=self.surrogate_type,
        )

        surrogate.fit(
            self.data[self.input_symbols].to_numpy(),
            self.data[self.output_symbols].to_numpy(),
        )

        print("Owen surrogate ready.")

        return surrogate

    def _normalize_targets(
        self,
        targets: list[str],
    ) -> list[str]:
        """Convert objective symbols such as f_1 to s_f_1."""
        normalized = []

        for target in targets:
            target = target.strip()

            if target in self.output_symbols:
                normalized_target = target

            elif target.startswith("f_"):
                normalized_target = f"s_{target}"

            else:
                raise ValueError(
                    f"Unknown target objective '{target}'."
                )

            if normalized_target not in self.output_symbols:
                raise ValueError(
                    f"Target '{target}' is not one of the problem "
                    f"objectives: {self.output_symbols}."
                )

            if normalized_target not in normalized:
                normalized.append(normalized_target)

        if not normalized:
            raise ValueError(
                "At least one target objective must be selected."
            )

        return normalized

    def explain(
        self,
        *,
        reference_point_original,
        solution_min,
        targets,
        weights: list[float] | None = None,
    ) -> dict:
        """Generate a serializable Owen/R-XIMO explanation."""
        target_symbols = self._normalize_targets(targets)

        result = generate_explanation_result(
            problem=self.problem,
            data=self.data,
            input_symbols=self.input_symbols,
            output_symbols=self.output_symbols,
            target_symbols=target_symbols,
            weights=weights,
            reference_point_original=np.asarray(
                reference_point_original,
                dtype=float,
            ),
            solution_min=np.asarray(
                solution_min,
                dtype=float,
            ),
            background_size=self.background_size,
            seed=self.seed,
            surrogate_model=self.surrogate,
            coalition_advantage_threshold=(
                self.coalition_advantage_threshold
            ),
        )

        suggestion = result.suggestion

        return {
            "type": "owen_explanation",
            "targets": [
                clean_symbol(target)
                for target in result.target_symbols
            ],
            "case_number": suggestion.case_number,
            "case_name": suggestion.case_name,
            "reference_point_status": (
                suggestion.reference_point_status
            ),
            "target_members": list(
                suggestion.target_members
            ),
            "actionable_target_members": list(
                suggestion.actionable_target_members
            ),
            "non_actionable_target_members": list(
                suggestion.non_actionable_target_members
            ),

            "rival_members": list(
                suggestion.rival_members
            ),
            "selected_action_rival_members": list(
                suggestion.selected_action_rival_members
            ),
            "actionable_rival_members": list(
                suggestion.actionable_rival_members
            ),
            "non_actionable_rival_members": list(
                suggestion.non_actionable_rival_members
            ),
            "strongest_supporter_members": list(
                suggestion.strongest_supporter_members
            ),
            "strongest_rival_members": list(
                suggestion.strongest_rival_members
            ),
            "explanation": suggestion.explanation,
            "suggestion": suggestion.suggestion,
            "owen_values": (
                result.owen_summary.to_dicts()
            ),
            "coalition_contributions": (
                result.coalition_summary.to_dicts()
            ),
        }
