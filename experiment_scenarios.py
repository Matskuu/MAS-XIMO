"""Print reproducible MAS-XIMO reference-point and target-set scenarios."""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from explanations.utils import get_objective_symbols
from problem_setup import PROBLEM_CHOICES, create_problem


PROFILE_NAMES = (
    "near_ideal",
    "balanced",
    "near_nadir",
    "mixed_a",
    "mixed_b",
)


def reference_profiles(problem) -> dict[str, np.ndarray]:
    """Construct normalized reference-point profiles for a DESDEO problem."""
    ideal = np.asarray(
        [float(objective.ideal) for objective in problem.objectives],
        dtype=float,
    )
    nadir = np.asarray(
        [float(objective.nadir) for objective in problem.objectives],
        dtype=float,
    )
    n_objectives = len(problem.objectives)

    fractions = {
        "near_ideal": np.full(n_objectives, 0.10),
        "balanced": np.full(n_objectives, 0.50),
        "near_nadir": np.full(n_objectives, 0.90),
        "mixed_a": np.asarray(
            [0.15 if index % 2 == 0 else 0.85 for index in range(n_objectives)],
            dtype=float,
        ),
    }
    fractions["mixed_b"] = 1.0 - fractions["mixed_a"]

    return {
        name: ideal + fraction * (nadir - ideal)
        for name, fraction in fractions.items()
    }


def default_target_sets(objective_symbols: list[str]) -> list[list[str]]:
    """Return all singleton and pair target sets for an objective set."""
    target_sets = [[symbol] for symbol in objective_symbols]
    target_sets.extend(
        [list(pair) for pair in combinations(objective_symbols, 2)]
    )
    return target_sets

def print_scenarios(problem_name: str) -> None:
    problem = create_problem(problem_name)
    symbols = get_objective_symbols(problem)
    ideal = np.asarray([float(obj.ideal) for obj in problem.objectives])
    nadir = np.asarray([float(obj.nadir) for obj in problem.objectives])

    print(f"Problem: {problem_name}")
    print(f"Objectives: {', '.join(symbols)}")
    print(f"Ideal: {ideal.tolist()}")
    print(f"Nadir: {nadir.tolist()}")
    print()

    print("Reference-point profiles")
    print("-" * 72)
    for name, reference_point in reference_profiles(problem).items():
        print(
            f"{name:12s}: "
            + " ".join(f"{value:.8g}" for value in reference_point)
        )

    print()
    print("Recommended target sets")
    print("-" * 72)
    for targets in default_target_sets(symbols):
        label = "single" if len(targets) == 1 else "pair"
        print(f"{label:6s}: " + " ".join(targets))

    if len(symbols) > 3:
        # Include representative 3-target coalitions without exhaustively
        # enumerating every larger subset.
        print("triple: " + " ".join(symbols[:3]))
        print("triple: " + " ".join(symbols[-3:]))

    print("all   : " + " ".join(symbols))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=PROBLEM_CHOICES, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_scenarios(args.problem)
