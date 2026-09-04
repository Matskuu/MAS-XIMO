"""Runtime utilities for multi-target Owen explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from explanations.owen_explainer import (
    MultiTargetOwenExplainer,
    summarize_owen_values,
)
from explanations.rximo_cases import (
    RXIMOSuggestion,
    build_rximo_suggestion,
    coalition_rows_to_candidates,
    classify_reference_point_status,
)
from explanations.utils import (
    get_ideal_and_nadir_minimized,
    get_objective_symbols,
    orient_objectives_from_minimize,
    orient_objectives_to_minimize,
)


@dataclass(frozen=True)
class ExplanationResult:
    """Store a complete explanation for one reference point and target set."""

    suggestion: RXIMOSuggestion
    owen_summary: pl.DataFrame
    coalition_summary: pl.DataFrame

    reference_point_original: np.ndarray
    reference_point_min: np.ndarray

    solution_original: np.ndarray
    solution_min: np.ndarray

    target_symbols: tuple[str, ...]
    input_symbols: tuple[str, ...]
    output_symbols: tuple[str, ...]


def validate_vector_length(
    values,
    expected_length: int,
    name: str,
) -> None:
    """Validate the length of an objective vector."""
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, "
            f"but received {len(values)}."
        )


def validate_target_symbols(
    target_symbols: list[str],
    output_symbols: list[str],
) -> list[str]:
    """Validate target symbols against the surrogate outputs."""
    invalid_targets = [
        target
        for target in target_symbols
        if target not in output_symbols
    ]

    if invalid_targets:
        raise ValueError(
            "Unknown target symbols: "
            + ", ".join(invalid_targets)
        )

    if len(target_symbols) == 0:
        raise ValueError(
            "At least one target objective is required."
        )

    return target_symbols

def build_explainer(
    *,
    data,
    input_symbols,
    output_symbols,
    target_symbols,
    weights,
    background_size,
    seed,
    surrogate_model,
):
    """Construct a target-specific Owen explainer."""
    background = data.sample(
        n=min(background_size, data.height),
        seed=seed,
    )

    explainer = MultiTargetOwenExplainer(
        problem_data=data,
        input_symbols=input_symbols,
        output_symbols=output_symbols,
        target_symbols=target_symbols,
        weights=weights,
        minimize=True,
        normalize_targets=True,
        asf_rho=1e-6,
        surrogate_model=surrogate_model,
        random_state=seed,
        fit_surrogate=False,
    )

    explainer.setup(
        background_data=background,
    )

    return explainer

def generate_explanation_result(
    *,
    problem,
    data,
    input_symbols,
    output_symbols,
    target_symbols,
    weights,
    reference_point_original,
    solution_min,
    background_size,
    seed,
    surrogate_model,
    coalition_advantage_threshold,
):
    """Compute one complete multi-target Owen explanation."""
    reference_point_original = np.asarray(
        reference_point_original,
        dtype=float,
    ).reshape(-1)

    solution_min = np.asarray(
        solution_min,
        dtype=float,
    ).reshape(-1)

    validate_vector_length(
        reference_point_original,
        len(input_symbols),
        "Reference point",
    )

    validate_vector_length(
        solution_min,
        len(output_symbols),
        "Solution vector",
    )

    target_symbols = validate_target_symbols(
        target_symbols,
        output_symbols,
    )

    reference_point_min = orient_objectives_to_minimize(
        problem,
        reference_point_original,
    )

    ideal_min, _ = get_ideal_and_nadir_minimized(problem)

    objective_symbols = get_objective_symbols(problem)

    solution_original = orient_objectives_from_minimize(
        problem,
        solution_min,
    )

    explainer = build_explainer(
        data=data,
        input_symbols=input_symbols,
        output_symbols=output_symbols,
        target_symbols=target_symbols,
        weights=weights,
        background_size=background_size,
        seed=seed,
        surrogate_model=surrogate_model,
    )

    explained_point = pl.DataFrame(
        {
            symbol: [value]
            for symbol, value in zip(
                input_symbols,
                reference_point_min,
                strict=True,
            )
        }
    )

    explanation = explainer.explain_input(
        explained_point
    )

    owen_summary = summarize_owen_values(
        explanation=explanation,
        input_symbols=input_symbols,
    )

    coalition_summary = (
        explainer.evaluate_all_coalitions(
            point=reference_point_min,
        )
    )

    candidates = coalition_rows_to_candidates(
        coalition_summary,
        target_symbols,
    )

    reference_point_status = (
        classify_reference_point_status(
            reference_point_min,
            solution_min,
        )
    )

    suggestion = build_rximo_suggestion(
        reference_point_status,
        candidates,
        target_symbols,
        coalition_advantage_threshold,
        reference_point_min,
        ideal_min,
        objective_symbols,
    )

    return ExplanationResult(
        suggestion=suggestion,
        owen_summary=owen_summary,
        coalition_summary=coalition_summary,
        reference_point_original=(
            reference_point_original
        ),
        reference_point_min=reference_point_min,
        solution_original=solution_original,
        solution_min=solution_min,
        target_symbols=tuple(target_symbols),
        input_symbols=tuple(input_symbols),
        output_symbols=tuple(output_symbols),
    )
