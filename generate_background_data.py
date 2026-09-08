"""Generate RPM background data for MAS-XIMO Owen explanations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from explanations.utils import (
    get_rximo_symbols,
    orient_objectives_to_minimize,
    solve_reference_point_with_desdeo_rpm,
)
from problem_setup import (
    PROBLEM_CHOICES,
    create_problem,
    default_background_path,
)


def sample_reference_point(problem, rng: np.random.Generator) -> np.ndarray:
    """Sample one reference point uniformly inside the ideal--nadir box."""
    values = []

    for objective in problem.objectives:
        ideal = float(objective.ideal)
        nadir = float(objective.nadir)
        lower = min(ideal, nadir)
        upper = max(ideal, nadir)
        values.append(rng.uniform(lower, upper))

    return np.asarray(values, dtype=float)


def generate_background_data(
    *,
    problem_name: str,
    samples: int,
    seed: int,
    output_path: Path,
    max_attempt_factor: int = 10,
) -> pl.DataFrame:
    """Generate successful reference-point/RPM-solution pairs."""
    problem = create_problem(problem_name)
    input_symbols, output_symbols = get_rximo_symbols(problem)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, float]] = []
    attempts = 0
    max_attempts = max(samples * max_attempt_factor, samples)

    while len(rows) < samples and attempts < max_attempts:
        attempts += 1
        reference_original = sample_reference_point(problem, rng)

        try:
            solution_min = solve_reference_point_with_desdeo_rpm(
                problem,
                reference_original,
            )
        except Exception as error:
            print(
                f"Skipping failed RPM solve {attempts}: {error}"
            )
            continue

        reference_min = orient_objectives_to_minimize(
            problem,
            reference_original,
        )

        row = {
            symbol: float(value)
            for symbol, value in zip(
                input_symbols,
                reference_min,
                strict=True,
            )
        }
        row.update(
            {
                symbol: float(value)
                for symbol, value in zip(
                    output_symbols,
                    solution_min,
                    strict=True,
                )
            }
        )
        rows.append(row)

        if len(rows) % 25 == 0 or len(rows) == samples:
            print(
                f"Generated {len(rows)}/{samples} successful RPM samples."
            )

    if len(rows) < samples:
        raise RuntimeError(
            f"Generated only {len(rows)} successful samples after "
            f"{attempts} attempts."
        )

    data = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_csv(output_path)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RPM background data for a MAS-XIMO problem."
    )
    parser.add_argument(
        "--problem",
        choices=PROBLEM_CHOICES,
        default="river",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or default_background_path(
        args.problem,
        seed=args.seed,
        samples=args.samples,
    )

    print(f"Problem: {args.problem}")
    print(f"Samples: {args.samples}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output}")

    generate_background_data(
        problem_name=args.problem,
        samples=args.samples,
        seed=args.seed,
        output_path=output,
    )


if __name__ == "__main__":
    main()
