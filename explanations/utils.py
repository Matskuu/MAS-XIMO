"""Shared utilities for DESDEO-based multi-target explanations.

This module contains reusable helpers for reading objective metadata,
converting objective vectors between their original orientations and a common
minimization orientation, and solving reference points with DESDEO's reference
point method.

The utilities are shared by background-data generation, the interactive
explanation workflow, and counterfactual testing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from desdeo.mcdm.reference_point_method import rpm_solve_solutions
from desdeo.problem import objective_dict_to_numpy_array


def get_objective_symbols(problem: Any) -> list[str]:
    """Return the objective symbols defined in a problem.

    Args:
        problem (Any): a DESDEO problem instance.

    Returns:
        list[str]: objective symbols in the order used by the problem.
    """
    return [
        str(objective.symbol)
        for objective in problem.objectives
    ]


def get_rximo_symbols(
    problem: Any,
) -> tuple[list[str], list[str]]:
    """Construct reference point and solution symbols for an R-XIMO dataset.

    Args:
        problem (Any): a DESDEO problem instance.

    Returns:
        tuple[list[str], list[str]]: reference point symbols prefixed with
        ``r_`` and solution symbols prefixed with ``s_``.
    """
    objective_symbols = get_objective_symbols(problem)

    input_symbols = [
        f"r_{symbol}"
        for symbol in objective_symbols
    ]
    output_symbols = [
        f"s_{symbol}"
        for symbol in objective_symbols
    ]

    return input_symbols, output_symbols


def get_ideal_and_nadir_minimized(
    problem: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal and nadir vectors in a common minimization orientation.

    Args:
        problem (Any): a DESDEO problem whose objectives define ideal, nadir,
        and maximization information.

    Returns:
        tuple[np.ndarray, np.ndarray]: ideal and nadir objective vectors in
        minimization orientation.
    """
    ideal_min = []
    nadir_min = []

    for objective in problem.objectives:
        ideal = float(objective.ideal)
        nadir = float(objective.nadir)

        if bool(getattr(objective, "maximize", False)):
            ideal_min.append(-ideal)
            nadir_min.append(-nadir)
        else:
            ideal_min.append(ideal)
            nadir_min.append(nadir)

    return (
        np.asarray(ideal_min, dtype=float),
        np.asarray(nadir_min, dtype=float),
    )


def orient_objectives_to_minimize(
    problem: Any,
    objective_vector_original: np.ndarray,
) -> np.ndarray:
    """Convert objective values to a common minimization orientation.

    Args:
        problem (Any): a DESDEO problem defining each objective direction.
        objective_vector_original (np.ndarray): objective values in their
            original maximization or minimization orientations.

    Returns:
        np.ndarray: objective values in minimization orientation.

    Raises:
        ValueError: if the vector length differs from the number of objectives.
    """
    objective_vector_original = np.asarray(
        objective_vector_original,
        dtype=float,
    ).reshape(-1)

    expected_length = len(problem.objectives)

    if objective_vector_original.size != expected_length:
        raise ValueError(
            "Objective vector must contain one value per problem objective. "
            f"Expected {expected_length}, but received "
            f"{objective_vector_original.size}."
        )

    values = []

    for value, objective in zip(
        objective_vector_original,
        problem.objectives,
        strict=True,
    ):
        if bool(getattr(objective, "maximize", False)):
            values.append(-float(value))
        else:
            values.append(float(value))

    return np.asarray(values, dtype=float)


def orient_objectives_from_minimize(
    problem: Any,
    objective_vector_min: np.ndarray,
) -> np.ndarray:
    """Convert objective values from minimization to original orientation.

    Args:
        problem (Any): a DESDEO problem defining each objective direction.
        objective_vector_min (np.ndarray): objective values in minimization
            orientation.

    Returns:
        np.ndarray: objective values in their original objective orientations.

    Raises:
        ValueError: if the vector length differs from the number of objectives.
    """
    objective_vector_min = np.asarray(
        objective_vector_min,
        dtype=float,
    ).reshape(-1)

    expected_length = len(problem.objectives)

    if objective_vector_min.size != expected_length:
        raise ValueError(
            "Objective vector must contain one value per problem objective. "
            f"Expected {expected_length}, but received "
            f"{objective_vector_min.size}."
        )

    values = []

    for value, objective in zip(
        objective_vector_min,
        problem.objectives,
        strict=True,
    ):
        if bool(getattr(objective, "maximize", False)):
            values.append(-float(value))
        else:
            values.append(float(value))

    return np.asarray(values, dtype=float)


def solve_reference_point_with_desdeo_rpm(
    problem: Any,
    reference_point_vector_original: np.ndarray,
    scalarization_options: dict | None = None,
    solver=None,
    solver_options: dict | None = None,
) -> np.ndarray:
    """Solve a reference point with DESDEO's reference point method.

    The reference point is supplied in the original objective orientation.
    The returned solution objective vector is converted to a common
    minimization orientation.

    Args:
        problem (Any): the multiobjective optimization problem to solve.
        reference_point_vector_original (np.ndarray): aspiration levels in
            original objective orientation and problem objective order.
        scalarization_options (dict | None, optional): keyword arguments passed
            to the achievement scalarizing function. Defaults to None.
        solver: optional solver initializer accepted by
            ``rpm_solve_solutions``.
        solver_options (dict | None, optional): options passed to the solver.
            Defaults to None.

    Returns:
        np.ndarray: the first RPM solution objective vector in minimization
        orientation.

    Raises:
        ValueError: if the reference point has an invalid length or contains
            non-finite values.
        RuntimeError: if RPM returns no usable solution or invalid objective
            values.
    """
    reference_point_vector_original = np.asarray(
        reference_point_vector_original,
        dtype=float,
    ).reshape(-1)

    objective_symbols = get_objective_symbols(problem)
    expected_length = len(objective_symbols)

    if reference_point_vector_original.size != expected_length:
        raise ValueError(
            "Reference point must contain one value per problem objective. "
            f"Expected {expected_length}, but received "
            f"{reference_point_vector_original.size}."
        )

    if not np.all(np.isfinite(reference_point_vector_original)):
        raise ValueError(
            "Reference point values must be finite."
        )

    reference_point = {
        symbol: float(value)
        for symbol, value in zip(
            objective_symbols,
            reference_point_vector_original,
            strict=True,
        )
    }

    results = rpm_solve_solutions(
        problem=problem,
        reference_point=reference_point,
        scalarization_options=scalarization_options,
        solver=solver,
        solver_options=solver_options,
    )

    if results is None or len(results) == 0:
        raise RuntimeError(
            "DESDEO RPM returned no solutions for the reference point."
        )

    first_result = results[0]

    if not hasattr(first_result, "optimal_objectives"):
        raise RuntimeError(
            "DESDEO RPM returned a result without optimal objective values."
        )

    if first_result.optimal_objectives is None:
        raise RuntimeError(
            "DESDEO RPM returned no optimal objective values."
        )

    objective_vector_original = objective_dict_to_numpy_array(
        problem,
        first_result.optimal_objectives,
    )

    objective_vector_original = np.asarray(
        objective_vector_original,
        dtype=float,
    ).reshape(-1)

    if objective_vector_original.size != expected_length:
        raise RuntimeError(
            "DESDEO RPM returned an objective vector with an unexpected "
            "number of values. "
            f"Expected {expected_length}, but received "
            f"{objective_vector_original.size}."
        )

    if not np.all(np.isfinite(objective_vector_original)):
        raise RuntimeError(
            "DESDEO RPM returned non-finite objective values."
        )

    return orient_objectives_to_minimize(
        problem,
        objective_vector_original,
    )

def validate_background_data(
    data: pl.DataFrame,
    *,
    input_symbols: list[str],
    output_symbols: list[str],
    expected_rows: int | None = None,
    source: str = "background dataset",
) -> None:
    """Validate a reference point--solution background dataset.

    Args:
        data: background dataset to validate.
        input_symbols: required reference point columns.
        output_symbols: required solution objective columns.
        expected_rows: optional required number of rows.
        source: human-readable source name used in error messages.

    Raises:
        ValueError: if required columns are missing, the dataset is empty,
            the row count is incorrect, or required values are nonnumeric or
            non-finite.
    """
    expected_columns = input_symbols + output_symbols

    missing_columns = [
        column
        for column in expected_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source} is missing required columns: "
            + ", ".join(missing_columns)
            + "."
        )

    if data.height == 0:
        raise ValueError(
            f"{source} contains no rows."
        )

    if (
        expected_rows is not None
        and data.height != expected_rows
    ):
        raise ValueError(
            f"{source} contains an unexpected number of rows. "
            f"Expected {expected_rows}, but found {data.height}."
        )

    try:
        values = (
            data.select(expected_columns)
            .to_numpy()
            .astype(float, copy=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source} contains nonnumeric values in required columns."
        ) from error

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{source} contains non-finite values in required columns."
        )
