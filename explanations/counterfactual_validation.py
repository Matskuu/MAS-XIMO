"""Counterfactual validation utilities for Owen-based R-XIMO suggestions.

This module constructs counterfactual reference points from generated
preference-change suggestions and evaluates whether the resulting RPM
solutions improve the selected target objectives together.

The numerical adjustment follows the standalone Owen explainer's
counterfactual testing procedure: target aspirations are strengthened toward
the ideal point and rival aspirations are relaxed toward the nadir point by a
fixed fraction of each objective's ideal--nadir range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from explanations.utils import (
    get_ideal_and_nadir_minimized,
    orient_objectives_from_minimize,
    orient_objectives_to_minimize,
)


CounterfactualStatus = Literal[
    "joint_improvement",
    "partial_improvement",
    "target_conflict",
    "no_improvement",
    "negligible_change",
]


@dataclass(frozen=True)
class TargetChange:
    """Describe the observed change of one target objective."""

    symbol: str
    original_value: float
    adjusted_value: float
    signed_improvement: float
    status: Literal[
        "improved",
        "worsened",
        "unchanged",
    ]


@dataclass(frozen=True)
class CounterfactualOutcome:
    """Store the result of counterfactual validation."""

    changes: tuple[TargetChange, ...]
    improved_targets: tuple[str, ...]
    worsened_targets: tuple[str, ...]
    unchanged_targets: tuple[str, ...]
    outcome_type: CounterfactualStatus
    adjusted_reference_point: np.ndarray
    adjusted_solution_min: np.ndarray


def clean_symbol(symbol: str) -> str:
    """Remove a reference-point or solution prefix from a symbol."""
    return (
        symbol
        .removeprefix("r_")
        .removeprefix("s_")
    )


def build_counterfactual_reference_point(
    problem,
    reference_point_original: np.ndarray,
    input_symbols: list[str],
    target_members: list[str] | tuple[str, ...],
    rival_members: list[str] | tuple[str, ...],
    step_fraction: float = 0.10,
) -> np.ndarray:
    """Construct a counterfactual reference point.

    Target aspirations are strengthened toward the ideal point and rival
    aspirations are relaxed toward the nadir point. Each change is scaled by
    the ideal--nadir range of the corresponding objective.

    Args:
        problem: DESDEO multiobjective optimization problem.
        reference_point_original: current reference point in original
            objective orientation.
        input_symbols: reference-point component symbols, such as
            ``["r_f_1", "r_f_2", ...]``.
        target_members: target reference-point components to strengthen.
        rival_members: rival reference-point components to relax.
        step_fraction: fraction of each ideal--nadir range used for the
            adjustment. Defaults to 0.10.

    Returns:
        Counterfactual reference point in original objective orientation.
    """
    if not 0.0 <= step_fraction <= 1.0:
        raise ValueError(
            "step_fraction must be between 0 and 1."
        )

    reference_point_original = np.asarray(
        reference_point_original,
        dtype=float,
    ).reshape(-1)

    if reference_point_original.size != len(
        problem.objectives
    ):
        raise ValueError(
            "Reference point must contain one value per objective."
        )

    reference_point_min = orient_objectives_to_minimize(
        problem,
        reference_point_original,
    )

    ideal_min, nadir_min = (
        get_ideal_and_nadir_minimized(problem)
    )

    low = np.minimum(
        ideal_min,
        nadir_min,
    )
    high = np.maximum(
        ideal_min,
        nadir_min,
    )
    ranges = high - low

    adjusted_reference_min = (
        reference_point_min.copy()
    )

    # Strengthen target aspirations toward the ideal.
    for member in target_members:
        if member not in input_symbols:
            raise ValueError(
                f"Unknown target member '{member}'."
            )

        index = input_symbols.index(member)

        adjusted_reference_min[index] = max(
            adjusted_reference_min[index]
            - step_fraction * ranges[index],
            low[index],
        )

    # Relax rival aspirations toward the nadir.
    for member in rival_members:
        if member not in input_symbols:
            raise ValueError(
                f"Unknown rival member '{member}'."
            )

        index = input_symbols.index(member)

        adjusted_reference_min[index] = min(
            adjusted_reference_min[index]
            + step_fraction * ranges[index],
            high[index],
        )

    return np.asarray(
        orient_objectives_from_minimize(
            problem,
            adjusted_reference_min,
        ),
        dtype=float,
    )


def evaluate_counterfactual_solution(
    problem,
    original_solution_min: np.ndarray,
    adjusted_solution_min: np.ndarray,
    target_symbols: list[str],
    adjusted_reference_point: np.ndarray,
    tolerance: float = 1e-6,
) -> CounterfactualOutcome:
    """Evaluate a counterfactual RPM solution for selected targets.

    The original and adjusted solutions are compared objective by objective.
    Objective direction is respected when determining whether each selected
    target improved, worsened, or remained effectively unchanged.

    Args:
        problem: DESDEO multiobjective optimization problem.
        original_solution_min: RPM solution for the original reference point
            in minimization orientation.
        adjusted_solution_min: RPM solution for the counterfactual reference
            point in minimization orientation.
        target_symbols: selected target objectives. Symbols may be supplied
            as ``f_i`` or ``s_f_i``.
        adjusted_reference_point: counterfactual reference point in original
            objective orientation.
        tolerance: relative threshold for classifying changes.

    Returns:
        Counterfactual validation result.
    """
    original_solution_min = np.asarray(
        original_solution_min,
        dtype=float,
    ).reshape(-1)

    adjusted_solution_min = np.asarray(
        adjusted_solution_min,
        dtype=float,
    ).reshape(-1)

    n_objectives = len(problem.objectives)

    if original_solution_min.size != n_objectives:
        raise ValueError(
            "Original solution must contain one value per objective."
        )

    if adjusted_solution_min.size != n_objectives:
        raise ValueError(
            "Adjusted solution must contain one value per objective."
        )

    if not np.all(
        np.isfinite(original_solution_min)
    ):
        raise ValueError(
            "Original solution must contain only finite values."
        )

    if not np.all(
        np.isfinite(adjusted_solution_min)
    ):
        raise ValueError(
            "Adjusted solution must contain only finite values."
        )

    objective_symbols = [
        str(objective.symbol)
        for objective in problem.objectives
    ]

    changes: list[TargetChange] = []

    for target in target_symbols:
        symbol = clean_symbol(target)

        if symbol not in objective_symbols:
            raise ValueError(
                f"Unknown target objective '{target}'."
            )

        index = objective_symbols.index(symbol)
        objective = problem.objectives[index]

        original_min = original_solution_min[index]
        adjusted_min = adjusted_solution_min[index]

        maximize = bool(
            getattr(
                objective,
                "maximize",
                False,
            )
        )

        if maximize:
            original_value = -original_min
            adjusted_value = -adjusted_min

            signed_improvement = (
                adjusted_value
                - original_value
            )

        else:
            original_value = original_min
            adjusted_value = adjusted_min

            signed_improvement = (
                original_value
                - adjusted_value
            )

        scale = max(
            abs(original_value),
            1.0,
        )

        relative_improvement = (
            signed_improvement / scale
        )

        if relative_improvement > tolerance:
            status = "improved"

        elif relative_improvement < -tolerance:
            status = "worsened"

        else:
            status = "unchanged"

        changes.append(
            TargetChange(
                symbol=symbol,
                original_value=float(
                    original_value
                ),
                adjusted_value=float(
                    adjusted_value
                ),
                signed_improvement=float(
                    signed_improvement
                ),
                status=status,
            )
        )

    improved = tuple(
        change.symbol
        for change in changes
        if change.status == "improved"
    )

    worsened = tuple(
        change.symbol
        for change in changes
        if change.status == "worsened"
    )

    unchanged = tuple(
        change.symbol
        for change in changes
        if change.status == "unchanged"
    )

    if len(improved) == len(changes):
        outcome_type = "joint_improvement"

    elif improved and worsened:
        outcome_type = "target_conflict"

    elif improved and unchanged:
        outcome_type = "partial_improvement"

    elif unchanged and not worsened:
        outcome_type = "negligible_change"

    else:
        outcome_type = "no_improvement"

    return CounterfactualOutcome(
        changes=tuple(changes),
        improved_targets=improved,
        worsened_targets=worsened,
        unchanged_targets=unchanged,
        outcome_type=outcome_type,
        adjusted_reference_point=np.asarray(
            adjusted_reference_point,
            dtype=float,
        ).copy(),
        adjusted_solution_min=(
            adjusted_solution_min.copy()
        ),
    )


def outcome_to_dict(
    outcome: CounterfactualOutcome,
) -> dict:
    """Convert a counterfactual outcome into a JSON-serializable dictionary."""
    return {
        "outcome_type": outcome.outcome_type,
        "joint_target_improvement": (
            outcome.outcome_type
            == "joint_improvement"
        ),
        "improved_targets": list(
            outcome.improved_targets
        ),
        "worsened_targets": list(
            outcome.worsened_targets
        ),
        "unchanged_targets": list(
            outcome.unchanged_targets
        ),
        "changes": [
            {
                "symbol": change.symbol,
                "original_value": (
                    change.original_value
                ),
                "adjusted_value": (
                    change.adjusted_value
                ),
                "signed_improvement": (
                    change.signed_improvement
                ),
                "status": change.status,
            }
            for change in outcome.changes
        ],
        "adjusted_reference_point": (
            outcome.adjusted_reference_point.tolist()
        ),
        "adjusted_solution_min": (
            outcome.adjusted_solution_min.tolist()
        ),
    }
