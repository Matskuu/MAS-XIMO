"""Print reproducible MAS-XIMO reference-point and target-set scenarios."""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from explanations.utils import get_objective_symbols
from problem_setup import PROBLEM_CHOICES, create_problem


PROFILES = {
    "near_ideal": 0.10,
    "balanced": 0.50,
    "near_nadir": 0.90,
}


def interpolate(ideal: np.ndarray, nadir: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate from ideal (0) to nadir (1) in original orientation."""
    return ideal + fraction * (nadir - ideal)


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
    for name, fraction in PROFILES.items():
        rp = interpolate(ideal, nadir, fraction)
        print(f"{name:12s}: " + " ".join(f"{x:.8g}" for x in rp))

    # Mixed profiles use alternating aspiration strength to create clearly
    # asymmetric situations that are useful for R-XIMO case exploration.
    if len(symbols) >= 3:
        fractions_a = np.asarray(
            [0.15 if i % 2 == 0 else 0.85 for i in range(len(symbols))]
        )
        fractions_b = 1.0 - fractions_a
        mixed_a = ideal + fractions_a * (nadir - ideal)
        mixed_b = ideal + fractions_b * (nadir - ideal)
        print("mixed_A     : " + " ".join(f"{x:.8g}" for x in mixed_a))
        print("mixed_B     : " + " ".join(f"{x:.8g}" for x in mixed_b))

    print()
    print("Recommended target sets")
    print("-" * 72)
    for symbol in symbols:
        print(f"single: {symbol}")
    for pair in combinations(symbols, 2):
        print("pair  : " + " ".join(pair))

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
