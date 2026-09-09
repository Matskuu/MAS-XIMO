"""Validate the MAS-XIMO Owen surrogate and explanations on DESDEO problems.

The validation covers three complementary properties:

1. Global surrogate fidelity on held-out RPM background samples.
2. Local surrogate fidelity on fresh RPM solves around selected reference points.
3. Owen decomposition fidelity, i.e. whether the SHAP/Owen base value plus
   attributions reconstructs the surrogate target-coalition utility.

The script supports the same ``river`` and ``spanish`` problem identifiers as
``mas_ximo.py`` and uses the same background-data naming convention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from explanations.owen_explainer import MultiTargetOwenExplainer, make_default_surrogate
from explanations.utils import (
    get_ideal_and_nadir_minimized,
    get_rximo_symbols,
    orient_objectives_from_minimize,
    orient_objectives_to_minimize,
    solve_reference_point_with_desdeo_rpm,
    validate_background_data,
)
from problem_setup import PROBLEM_CHOICES, create_problem, default_background_path


@dataclass(frozen=True)
class ObjectiveFidelityMetrics:
    """Surrogate fidelity metrics for one objective."""

    mean_mae: float
    max_mae: float
    mean_normalized_mae: float
    max_normalized_mae: float
    correlation: float
    actual_std: float
    predicted_std: float
    actual_range: float
    predicted_range: float


@dataclass(frozen=True)
class FidelityMetrics:
    """Summary metrics for surrogate predictions."""

    mean_solution_mae: float
    max_solution_mae: float
    mean_normalized_solution_mae: float
    max_normalized_solution_mae: float
    mean_utility_abs_error: float
    max_utility_abs_error: float
    utility_correlation: float
    actual_utility_std: float
    predicted_utility_std: float
    actual_utility_range: float
    predicted_utility_range: float
    objective_metrics: tuple[ObjectiveFidelityMetrics, ...]


def normalize_targets(raw_targets: list[str], output_symbols: list[str]) -> list[str]:
    """Map problem objective symbols to surrogate output symbols."""
    normalized: list[str] = []

    for target in raw_targets:
        target = target.strip()
        if target in output_symbols:
            symbol = target
        elif f"s_{target}" in output_symbols:
            symbol = f"s_{target}"
        else:
            raise ValueError(
                f"Unknown target '{target}'. Available objectives are: "
                + ", ".join(symbol.removeprefix("s_") for symbol in output_symbols)
            )

        if symbol not in normalized:
            normalized.append(symbol)

    if not normalized:
        raise ValueError("At least one target objective is required.")

    return normalized


def target_utility_from_outputs(
    explainer: MultiTargetOwenExplainer,
    outputs: np.ndarray,
) -> np.ndarray:
    """Evaluate the explainer's target utility from supplied objective vectors."""
    outputs = np.atleast_2d(np.asarray(outputs, dtype=float))
    targets = outputs[:, explainer.target_indices]

    if explainer.normalize_targets:
        targets = (targets - explainer.target_mins) / explainer.target_ranges

    weighted_targets = targets * explainer.weights
    asf_value = np.max(weighted_targets, axis=1) + explainer.asf_rho * np.sum(
        weighted_targets, axis=1
    )
    return np.asarray(-asf_value if explainer.minimize else asf_value, dtype=float)


def objective_ranges(problem) -> np.ndarray:
    """Return ideal--nadir ranges for normalized objective-error metrics.

    Using the problem's objective ranges keeps the normalization meaningful
    when RPM solutions are nearly constant in one objective. In particular,
    it avoids dividing tiny prediction errors by an even smaller observed
    solution range.
    """
    ideal_min, nadir_min = get_ideal_and_nadir_minimized(problem)
    ranges = np.abs(np.asarray(nadir_min, dtype=float) - np.asarray(ideal_min, dtype=float))
    return np.where(np.isclose(ranges, 0.0), 1.0, ranges)


def correlation_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation when both vectors have nonzero variance."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size < 2 or np.isclose(np.std(a), 0.0) or np.isclose(np.std(b), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_fidelity_metrics(
    *,
    explainer: MultiTargetOwenExplainer,
    actual_outputs: np.ndarray,
    predicted_outputs: np.ndarray,
    output_ranges: np.ndarray,
) -> FidelityMetrics:
    """Compute objective-vector and target-utility fidelity metrics."""
    actual_outputs = np.asarray(actual_outputs, dtype=float)
    predicted_outputs = np.asarray(predicted_outputs, dtype=float)

    abs_error = np.abs(predicted_outputs - actual_outputs)
    normalized_abs_error = abs_error / output_ranges.reshape(1, -1)

    actual_utility = target_utility_from_outputs(explainer, actual_outputs)
    predicted_utility = target_utility_from_outputs(explainer, predicted_outputs)
    utility_error = np.abs(predicted_utility - actual_utility)

    per_objective = tuple(
        ObjectiveFidelityMetrics(
            mean_mae=float(np.mean(abs_error[:, index])),
            max_mae=float(np.max(abs_error[:, index])),
            mean_normalized_mae=float(np.mean(normalized_abs_error[:, index])),
            max_normalized_mae=float(np.max(normalized_abs_error[:, index])),
            correlation=correlation_or_nan(
                actual_outputs[:, index], predicted_outputs[:, index]
            ),
            actual_std=float(np.std(actual_outputs[:, index])),
            predicted_std=float(np.std(predicted_outputs[:, index])),
            actual_range=float(np.ptp(actual_outputs[:, index])),
            predicted_range=float(np.ptp(predicted_outputs[:, index])),
        )
        for index in range(actual_outputs.shape[1])
    )

    return FidelityMetrics(
        mean_solution_mae=float(np.mean(abs_error)),
        max_solution_mae=float(np.max(abs_error)),
        mean_normalized_solution_mae=float(np.mean(normalized_abs_error)),
        max_normalized_solution_mae=float(np.max(normalized_abs_error)),
        mean_utility_abs_error=float(np.mean(utility_error)),
        max_utility_abs_error=float(np.max(utility_error)),
        utility_correlation=correlation_or_nan(actual_utility, predicted_utility),
        actual_utility_std=float(np.std(actual_utility)),
        predicted_utility_std=float(np.std(predicted_utility)),
        actual_utility_range=float(np.ptp(actual_utility)),
        predicted_utility_range=float(np.ptp(predicted_utility)),
        objective_metrics=per_objective,
    )


def split_indices(n: int, test_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/test indices."""
    if n < 2:
        raise ValueError("At least two background samples are required.")
    test_size = min(max(1, test_size), n - 1)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    return indices[test_size:], indices[:test_size]


def build_validation_explainer(
    *,
    train_data: pl.DataFrame,
    input_symbols: list[str],
    output_symbols: list[str],
    target_symbols: list[str],
    surrogate_type: str,
    gp_config: str,
    seed: int,
    background_size: int,
) -> MultiTargetOwenExplainer:
    """Fit the validation surrogate and configure its partition explainer."""
    surrogate = make_default_surrogate(
        random_state=seed,
        surrogate_type=surrogate_type,
        gp_config=gp_config,
    )
    surrogate.fit(
        train_data[input_symbols].to_numpy(),
        train_data[output_symbols].to_numpy(),
    )

    explainer = MultiTargetOwenExplainer(
        problem_data=train_data,
        input_symbols=input_symbols,
        output_symbols=output_symbols,
        target_symbols=target_symbols,
        weights=None,
        minimize=True,
        normalize_targets=True,
        asf_rho=1e-6,
        surrogate_model=surrogate,
        random_state=seed,
        fit_surrogate=False,
    )

    background = train_data.sample(
        n=min(background_size, train_data.height),
        seed=seed,
    )
    explainer.setup(background_data=background)
    return explainer


def profile_reference_point(problem, fractions: list[float]) -> np.ndarray:
    """Construct an original-orientation RP by interpolating ideal to nadir."""
    if len(fractions) != len(problem.objectives):
        raise ValueError("Profile must contain one fraction per objective.")

    values = []
    for objective, fraction in zip(problem.objectives, fractions, strict=True):
        ideal = float(objective.ideal)
        nadir = float(objective.nadir)
        values.append(ideal + float(fraction) * (nadir - ideal))
    return np.asarray(values, dtype=float)


def default_profiles(problem) -> dict[str, np.ndarray]:
    """Return balanced and asymmetric test reference points."""
    n = len(problem.objectives)
    return {
        "near_ideal": profile_reference_point(problem, [0.10] * n),
        "balanced": profile_reference_point(problem, [0.50] * n),
        "near_nadir": profile_reference_point(problem, [0.90] * n),
        "mixed_a": profile_reference_point(
            problem, [0.15 if i % 2 == 0 else 0.85 for i in range(n)]
        ),
        "mixed_b": profile_reference_point(
            problem, [0.85 if i % 2 == 0 else 0.15 for i in range(n)]
        ),
    }


def perturb_reference_points(
    *,
    problem,
    center_original: np.ndarray,
    count: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Generate clipped local perturbations in original objective orientation."""
    rng = np.random.default_rng(seed)
    ideal = np.asarray([float(obj.ideal) for obj in problem.objectives], dtype=float)
    nadir = np.asarray([float(obj.nadir) for obj in problem.objectives], dtype=float)
    scale = np.abs(nadir - ideal)
    lower = np.minimum(ideal, nadir)
    upper = np.maximum(ideal, nadir)

    noise = rng.normal(0.0, fraction, size=(count, center_original.size)) * scale
    points = center_original.reshape(1, -1) + noise
    return np.clip(points, lower, upper)


def run_local_validation(
    *,
    problem,
    explainer: MultiTargetOwenExplainer,
    centers: dict[str, np.ndarray],
    local_samples: int,
    perturbation_fraction: float,
    seed: int,
    output_ranges: np.ndarray,
) -> list[tuple[str, int, FidelityMetrics]]:
    """Validate the surrogate on fresh RPM solves around reference points."""
    rows: list[tuple[str, int, FidelityMetrics]] = []

    for offset, (name, center) in enumerate(centers.items()):
        points_original = perturb_reference_points(
            problem=problem,
            center_original=center,
            count=local_samples,
            fraction=perturbation_fraction,
            seed=seed + offset + 1,
        )

        actual_outputs = []
        valid_inputs_min = []
        failed = 0

        for point_original in points_original:
            try:
                solution_min = solve_reference_point_with_desdeo_rpm(problem, point_original)
            except Exception as error:
                failed += 1
                print(f"  Local RPM solve failed near {name}: {error}")
                continue

            valid_inputs_min.append(orient_objectives_to_minimize(problem, point_original))
            actual_outputs.append(solution_min)

        if not actual_outputs:
            print(f"  {name}: no successful local RPM solves; skipped.")
            continue

        inputs = np.asarray(valid_inputs_min, dtype=float)
        actual = np.asarray(actual_outputs, dtype=float)
        predicted = explainer.evaluate(inputs)
        metrics = compute_fidelity_metrics(
            explainer=explainer,
            actual_outputs=actual,
            predicted_outputs=predicted,
            output_ranges=output_ranges,
        )
        rows.append((name, failed, metrics))

    return rows


def owen_reconstruction(
    *,
    problem,
    explainer: MultiTargetOwenExplainer,
    reference_points: dict[str, np.ndarray],
) -> list[tuple[str, float, float, float]]:
    """Check SHAP/Owen additive reconstruction at selected reference points."""
    rows = []

    for name, point_original in reference_points.items():
        point_min = orient_objectives_to_minimize(problem, point_original)
        frame = pl.DataFrame(
            {symbol: [value] for symbol, value in zip(explainer.input_symbols, point_min, strict=True)}
        )
        explanation = explainer.explain_input(frame)

        values = np.asarray(explanation.values, dtype=float).reshape(-1)
        base = float(np.asarray(explanation.base_values, dtype=float).reshape(-1)[0])
        reconstructed = base + float(np.sum(values))
        actual = float(explainer.evaluate_reference_point(point_min)[0])
        rows.append((name, actual, reconstructed, abs(actual - reconstructed)))

    return rows


def format_correlation(value: float) -> str:
    """Format correlation while making near-constant vectors explicit."""
    if np.isnan(value):
        return "n/a (near-constant)"
    return f"{value:.6g}"


def print_metrics(
    prefix: str,
    metrics: FidelityMetrics,
    objective_symbols: list[str],
) -> None:
    """Print aggregate, per-objective, and target-utility fidelity metrics."""
    print(prefix)
    print(f"  mean solution MAE:              {metrics.mean_solution_mae:.6g}")
    print(f"  max solution MAE:               {metrics.max_solution_mae:.6g}")
    print(f"  mean normalized solution MAE:   {metrics.mean_normalized_solution_mae:.6g}")
    print(f"  max normalized solution MAE:    {metrics.max_normalized_solution_mae:.6g}")

    print("  per-objective fidelity (normalized by ideal--nadir range):")
    for symbol, objective in zip(
        objective_symbols, metrics.objective_metrics, strict=True
    ):
        display_symbol = symbol.removeprefix("s_")
        print(
            f"    {display_symbol}: mean MAE={objective.mean_mae:.6g}, "
            f"mean normalized MAE={objective.mean_normalized_mae:.6g}, "
            f"correlation={format_correlation(objective.correlation)}"
        )
        print(
            f"       max MAE={objective.max_mae:.6g}, "
            f"max normalized MAE={objective.max_normalized_mae:.6g}"
        )
        print(
            f"       actual std={objective.actual_std:.6g}, "
            f"predicted std={objective.predicted_std:.6g}"
        )
        print(
            f"       actual range={objective.actual_range:.6g}, "
            f"predicted range={objective.predicted_range:.6g}"
        )

    print(f"  mean target-utility abs error:  {metrics.mean_utility_abs_error:.6g}")
    print(f"  max target-utility abs error:   {metrics.max_utility_abs_error:.6g}")
    print(
        "  target-utility correlation:     "
        f"{format_correlation(metrics.utility_correlation)}"
    )
    print(
        f"  target-utility actual std:      {metrics.actual_utility_std:.6g}"
    )
    print(
        f"  target-utility predicted std:   {metrics.predicted_utility_std:.6g}"
    )
    print(
        f"  target-utility actual range:    {metrics.actual_utility_range:.6g}"
    )
    print(
        f"  target-utility predicted range: {metrics.predicted_utility_range:.6g}"
    )



def _format_array(values) -> str:
    """Format scalar or array-valued learned hyperparameters compactly."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return f"{float(array):.6g}"
    return "[" + ", ".join(f"{value:.6g}" for value in array.reshape(-1)) + "]"


def print_surrogate_diagnostics(
    explainer: MultiTargetOwenExplainer,
    output_symbols: list[str],
    surrogate_type: str,
) -> None:
    """Print fitted-model diagnostics for each surrogate output.

    For Gaussian-process surrogates, this exposes the optimized kernel, its
    principal learned hyperparameters, the log-marginal likelihood, and the
    output normalization statistics. These values are useful for detecting
    degenerate fits in which an output model collapses to an almost constant
    predictor.
    """
    print("\nSURROGATE MODEL DIAGNOSTICS")
    print("-" * 60)

    if surrogate_type != "gaussian_process":
        print("  Detailed kernel diagnostics are only available for gaussian_process.")
        return

    model = explainer.surrogate_model
    estimators = getattr(model, "estimators_", None)
    if estimators is None:
        print("  Fitted per-output estimators are unavailable.")
        return

    for symbol, estimator in zip(output_symbols, estimators, strict=True):
        display_symbol = symbol.removeprefix("s_")
        named_steps = getattr(estimator, "named_steps", {})
        gp = named_steps.get("gaussianprocessregressor")
        scaler = named_steps.get("standardscaler")

        print(f"  {display_symbol}:")
        if gp is None:
            print(f"    estimator: {estimator!r}")
            continue

        kernel = getattr(gp, "kernel_", None)
        if kernel is None:
            print("    fitted kernel: unavailable")
            continue

        print(f"    fitted kernel: {kernel}")

        try:
            print(
                "    constant value: "
                f"{_format_array(kernel.k1.k1.constant_value)}"
            )
            print(
                "    RBF length scale: "
                f"{_format_array(kernel.k1.k2.length_scale)}"
            )
            print(
                "    white-noise level: "
                f"{_format_array(kernel.k2.noise_level)}"
            )
        except AttributeError:
            print(
                "    optimized theta (log-space): "
                f"{_format_array(kernel.theta)}"
            )

        lml = getattr(gp, "log_marginal_likelihood_value_", None)
        if lml is not None:
            print(f"    log-marginal likelihood: {float(lml):.6g}")

        y_mean = getattr(gp, "_y_train_mean", None)
        y_std = getattr(gp, "_y_train_std", None)
        if y_mean is not None and y_std is not None:
            print(
                "    y normalization: "
                f"mean={_format_array(y_mean)}, std={_format_array(y_std)}"
            )

        if scaler is not None and hasattr(scaler, "scale_"):
            print(
                "    input scaler scale: "
                f"{_format_array(scaler.scale_)}"
            )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Owen surrogate and explanation fidelity."
    )
    parser.add_argument("--problem", choices=PROBLEM_CHOICES, default="river")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Target objective symbols, e.g. f_1 f_3 or f1 f3.",
    )
    parser.add_argument(
        "--surrogate-type",
        choices=("gaussian_process", "random_forest"),
        default="gaussian_process",
    )
    parser.add_argument(
        "--gp-config",
        choices=("current", "bounded", "bounded_restarts"),
        default="current",
        help=(
            "Gaussian-process configuration. 'current' uses the existing "
            "length-scale bounds and no restarts; 'bounded' uses "
            "length_scale_bounds=(0.05, 100) with no restarts; "
            "'bounded_restarts' uses the same bounds with five restarts."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--local-samples", type=int, default=8)
    parser.add_argument("--local-fraction", type=float, default=0.05)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("near_ideal", "balanced", "near_nadir", "mixed_a", "mixed_b"),
        default=("near_ideal", "balanced", "near_nadir", "mixed_a", "mixed_b"),
    )
    parser.add_argument(
        "--background-data",
        type=Path,
        default=None,
        help="Optional explicit RPM background CSV path.",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip fresh local RPM solves.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = create_problem(args.problem)
    input_symbols, output_symbols = get_rximo_symbols(problem)

    default_raw_targets = (
        ["f_1", "f_3"] if args.problem == "river" else ["f1", "f3"]
    )
    raw_targets = args.targets or default_raw_targets
    target_symbols = normalize_targets(raw_targets, output_symbols)

    data_path = args.background_data or default_background_path(
        args.problem,
        seed=args.seed,
        samples=args.samples,
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Background dataset not found: {data_path}\n"
            "Start MAS-XIMO once for this problem to generate it automatically, "
            "or run generate_background_data.py."
        )

    data = pl.read_csv(data_path)
    validate_background_data(
        data,
        input_symbols=input_symbols,
        output_symbols=output_symbols,
        source=str(data_path),
    )

    train_idx, test_idx = split_indices(data.height, args.test_size, args.seed)
    indexed = data.with_row_index("__row_index")
    train_data = indexed.filter(
        pl.col("__row_index").is_in(train_idx.tolist())
    ).drop("__row_index")
    test_data = indexed.filter(
        pl.col("__row_index").is_in(test_idx.tolist())
    ).drop("__row_index")

    print("OWEN EXPLANATION VALIDATION")
    print("=" * 60)
    print(f"Problem:          {args.problem}")
    print(f"Dataset:          {data_path}")
    print(f"Targets:          {', '.join(raw_targets)}")
    print(f"Surrogate:        {args.surrogate_type}")
    if args.surrogate_type == "gaussian_process":
        print(f"GP config:        {args.gp_config}")
    print(f"Training samples: {train_data.height}")
    print(f"Held-out samples: {test_data.height}")
    print(f"SHAP background:  {min(args.background_size, train_data.height)}")

    explainer = build_validation_explainer(
        train_data=train_data,
        input_symbols=input_symbols,
        output_symbols=output_symbols,
        target_symbols=target_symbols,
        surrogate_type=args.surrogate_type,
        gp_config=args.gp_config,
        seed=args.seed,
        background_size=args.background_size,
    )
    print_surrogate_diagnostics(
        explainer=explainer,
        output_symbols=output_symbols,
        surrogate_type=args.surrogate_type,
    )

    ranges = objective_ranges(problem)

    test_inputs = test_data[input_symbols].to_numpy()
    test_actual = test_data[output_symbols].to_numpy()
    test_predicted = explainer.evaluate(test_inputs)
    global_metrics = compute_fidelity_metrics(
        explainer=explainer,
        actual_outputs=test_actual,
        predicted_outputs=test_predicted,
        output_ranges=ranges,
    )

    print("\nGLOBAL FIDELITY — HELD-OUT RPM SAMPLES")
    print("-" * 60)
    print_metrics("", global_metrics, output_symbols)

    all_profiles = default_profiles(problem)
    selected_profiles = {name: all_profiles[name] for name in args.profiles}

    print("\nREFERENCE POINTS USED FOR LOCAL/OWEN CHECKS")
    print("-" * 60)
    for name, point in selected_profiles.items():
        text = ", ".join(f"{value:.6g}" for value in point)
        print(f"  {name:12s}: [{text}]")

    if not args.skip_local:
        print("\nLOCAL FIDELITY — FRESH RPM SOLVES")
        print("-" * 60)
        local_rows = run_local_validation(
            problem=problem,
            explainer=explainer,
            centers=selected_profiles,
            local_samples=args.local_samples,
            perturbation_fraction=args.local_fraction,
            seed=args.seed,
            output_ranges=ranges,
        )
        for name, failed, metrics in local_rows:
            print_metrics(
                f"\n{name} (failed solves: {failed})", metrics, output_symbols
            )

    print("\nOWEN DECOMPOSITION FIDELITY")
    print("-" * 60)
    for name, actual, reconstructed, error in owen_reconstruction(
        problem=problem,
        explainer=explainer,
        reference_points=selected_profiles,
    ):
        print(
            f"  {name:12s}: utility={actual:+.8f}, "
            f"reconstructed={reconstructed:+.8f}, abs_error={error:.3e}"
        )

    ideal_min, nadir_min = get_ideal_and_nadir_minimized(problem)
    print("\nOBJECTIVE RANGE CONTEXT (original orientation)")
    print("-" * 60)
    ideal_original = orient_objectives_from_minimize(problem, ideal_min)
    nadir_original = orient_objectives_from_minimize(problem, nadir_min)
    for objective, ideal, nadir in zip(
        problem.objectives, ideal_original, nadir_original, strict=True
    ):
        print(f"  {objective.symbol}: ideal={ideal:.6g}, nadir={nadir:.6g}")


if __name__ == "__main__":
    main()
