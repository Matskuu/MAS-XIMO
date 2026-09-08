"""Problem configuration for MAS-XIMO experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from desdeo.problem.schema import TensorVariable
from desdeo.problem.testproblems import (
    river_pollution_problem,
    spanish_sustainability_problem,
)

PROBLEM_CHOICES = ("river", "spanish")


def create_problem(problem_name: str = "river"):
    """Create a supported DESDEO test problem.

    Args:
        problem_name: Short problem identifier. Supported values are
            ``"river"`` and ``"spanish"``.

    Returns:
        A DESDEO problem instance.

    Raises:
        ValueError: if ``problem_name`` is unknown.
    """
    normalized = problem_name.strip().lower()

    if normalized == "river":
        return river_pollution_problem(
            five_objective_variant=False,
        )

    if normalized == "spanish":
        problem = spanish_sustainability_problem()
        return _replace_invalid_initial_values(problem)

    raise ValueError(
        f"Unknown problem '{problem_name}'. "
        f"Choose one of: {', '.join(PROBLEM_CHOICES)}."
    )


def _replace_invalid_initial_values(problem):
    """Return a copy of a problem with bounded midpoint initial values.

    DESDEO's Spanish sustainability test problem currently initializes the
    complete decision vector ``X`` to 1.0. Several elements of ``X`` have
    lower bounds much larger than 1.0, which causes Pyomo W1002 warnings when
    an RPM scalarization is constructed. Using the midpoint of each finite
    bound interval gives every decision variable a valid starting value while
    leaving the problem formulation and bounds unchanged.
    """
    updated_variables = []

    for variable in problem.variables:
        if hasattr(variable, "get_lowerbound_values") and hasattr(
            variable, "get_upperbound_values"
        ):
            lower = np.asarray(variable.get_lowerbound_values(), dtype=float)
            upper = np.asarray(variable.get_upperbound_values(), dtype=float)

            if lower.shape == upper.shape and np.all(np.isfinite(lower)) and np.all(
                np.isfinite(upper)
            ):
                initial = ((lower + upper) / 2.0).tolist()

                # Reconstruct TensorVariable through its normal Pydantic
                # constructor. ``model_copy(update=...)`` does not validate
                # updated values, so assigning a raw Python list directly to
                # ``initial_values`` would bypass DESDEO's MathJSON parser.
                # That later causes ``get_tensor_values`` to reject the raw
                # list during scalarization.
                if isinstance(variable, TensorVariable):
                    variable = TensorVariable(
                        name=variable.name,
                        symbol=variable.symbol,
                        variable_type=variable.variable_type,
                        shape=variable.shape,
                        lowerbounds=variable.lowerbounds,
                        upperbounds=variable.upperbounds,
                        initial_values=initial,
                    )

        updated_variables.append(variable)

    return problem.model_copy(update={"variables": updated_variables})


def default_background_path(
    problem_name: str,
    *,
    seed: int = 1,
    samples: int = 300,
) -> Path:
    """Return the conventional background-data path for a problem."""
    base = Path(__file__).resolve().parent / "data"
    normalized = problem_name.strip().lower()

    if normalized == "river":
        filename = (
            f"river_rximo_rpm_background_4obj_"
            f"{samples}samples_seed{seed}.csv"
        )
    elif normalized == "spanish":
        filename = (
            f"spanish_rximo_rpm_background_3obj_"
            f"{samples}samples_seed{seed}.csv"
        )
    else:
        raise ValueError(
            f"Unknown problem '{problem_name}'. "
            f"Choose one of: {', '.join(PROBLEM_CHOICES)}."
        )

    return base / filename
