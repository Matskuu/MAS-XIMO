"""Evaluate Owen explanations across preference regions and target sets.

This experiment complements surrogate-fidelity validation by testing the
human-facing explanation pipeline itself. For each reference-point profile and
target set, the script solves the RPM problem, generates a generalized R-XIMO
Owen explanation, applies the suggested preference adjustment, solves the
counterfactual reference point, and records whether the selected targets
actually improve.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from explanations.counterfactual_validation import (
    build_counterfactual_reference_point,
    evaluate_counterfactual_solution,
    outcome_to_dict,
)
from explanations.owen_service import OwenExplanationService
from explanations.utils import (
    get_objective_symbols,
    orient_objectives_from_minimize,
    solve_reference_point_with_desdeo_rpm,
)
from generate_background_data import generate_background_data
from problem_setup import PROBLEM_CHOICES, create_problem, default_background_path


PROFILE_FRACTIONS = {
    "near_ideal": None,
    "balanced": None,
    "near_nadir": None,
    "mixed_a": None,
    "mixed_b": None,
}


def reference_profiles(problem) -> dict[str, np.ndarray]:
    """Return deterministic reference points spanning symmetric and mixed regions."""
    ideal = np.asarray([float(obj.ideal) for obj in problem.objectives], dtype=float)
    nadir = np.asarray([float(obj.nadir) for obj in problem.objectives], dtype=float)

    def interpolate(fraction):
        fraction = np.asarray(fraction, dtype=float)
        return ideal + fraction * (nadir - ideal)

    n_obj = len(problem.objectives)
    mixed_a = np.asarray([0.15 if i % 2 == 0 else 0.85 for i in range(n_obj)])
    mixed_b = 1.0 - mixed_a

    return {
        "near_ideal": interpolate(np.full(n_obj, 0.10)),
        "balanced": interpolate(np.full(n_obj, 0.50)),
        "near_nadir": interpolate(np.full(n_obj, 0.90)),
        "mixed_a": interpolate(mixed_a),
        "mixed_b": interpolate(mixed_b),
    }


def default_target_sets(objective_symbols: list[str]) -> list[list[str]]:
    """Use all singleton and pair target sets as the compact default sweep."""
    sets = [[symbol] for symbol in objective_symbols]
    sets.extend([list(pair) for pair in combinations(objective_symbols, 2)])
    return sets


def parse_target_sets(raw_sets: list[str] | None, objective_symbols: list[str]) -> list[list[str]]:
    """Parse comma-separated target sets supplied on the command line."""
    if not raw_sets:
        return default_target_sets(objective_symbols)

    parsed: list[list[str]] = []
    for raw in raw_sets:
        targets = [item.strip() for item in raw.split(",") if item.strip()]
        if not targets:
            raise ValueError(f"Empty target set: {raw!r}")
        unknown = [target for target in targets if target not in objective_symbols]
        if unknown:
            raise ValueError(
                f"Unknown targets {unknown}. Available objectives: {objective_symbols}"
            )
        parsed.append(targets)
    return parsed


def json_text(value) -> str:
    """Serialize nested explanation content safely for CSV output."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ensure_background(problem_name: str, samples: int, seed: int) -> Path:
    """Return the expected RPM background data path, generating it if missing."""
    path = default_background_path(problem_name, seed=seed, samples=samples)
    if path.exists():
        return path

    print(f"Background dataset not found. Generating {samples} RPM samples...")
    generate_background_data(
        problem_name=problem_name,
        samples=samples,
        seed=seed,
        output_path=path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Owen explanations, generalized R-XIMO suggestions, and "
            "counterfactual outcomes across reference-point regions."
        )
    )
    parser.add_argument("--problem", choices=PROBLEM_CHOICES, default="river")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument(
        "--surrogate-type",
        choices=("gaussian_process", "random_forest"),
        default="gaussian_process",
    )
    parser.add_argument(
        "--target-sets",
        nargs="*",
        default=None,
        metavar="TARGETS",
        help=(
            "Optional comma-separated target sets, e.g. --target-sets "
            "f_1 f_3 f_1,f_3. If omitted, all singleton and pair sets are tested."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_FRACTIONS),
        default=list(PROFILE_FRACTIONS),
    )
    parser.add_argument("--counterfactual-step", type=float, default=0.10)
    parser.add_argument("--coalition-advantage-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    args = parser.parse_args()

    problem = create_problem(args.problem)
    objective_symbols = get_objective_symbols(problem)
    profiles = reference_profiles(problem)
    target_sets = parse_target_sets(args.target_sets, objective_symbols)
    background_path = ensure_background(args.problem, args.samples, args.seed)

    service = OwenExplanationService(
        problem=problem,
        background_data_path=background_path,
        surrogate_type=args.surrogate_type,
        seed=args.seed,
        background_size=args.background_size,
        coalition_advantage_threshold=args.coalition_advantage_threshold,
    )

    print("OWEN EXPLANATION BEHAVIOR EVALUATION")
    print("=" * 68)
    print(f"Problem:             {args.problem}")
    print(f"Background samples:  {args.samples}")
    print(f"Surrogate:           {args.surrogate_type}")
    print(f"Target sets:         {len(target_sets)}")
    print(f"Profiles:            {', '.join(args.profiles)}")
    print(f"Counterfactual step: {args.counterfactual_step:.3f}")

    rows: list[dict] = []

    for targets in target_sets:
        target_label = "+".join(targets)
        print(f"\nTargets: {target_label}")
        print("-" * 68)

        for profile_name in args.profiles:
            reference_original = profiles[profile_name]
            try:
                solution_min = solve_reference_point_with_desdeo_rpm(
                    problem,
                    reference_original,
                )

                explanation = service.explain(
                    reference_point_original=reference_original,
                    solution_min=solution_min,
                    targets=targets,
                )

                actionable_targets = explanation["actionable_target_members"]
                selected_rivals = explanation["selected_action_rival_members"]

                adjusted_reference = build_counterfactual_reference_point(
                    problem,
                    reference_original,
                    service.input_symbols,
                    actionable_targets,
                    selected_rivals,
                    step_fraction=args.counterfactual_step,
                )
                adjusted_solution_min = solve_reference_point_with_desdeo_rpm(
                    problem,
                    adjusted_reference,
                )
                outcome = evaluate_counterfactual_solution(
                    problem,
                    solution_min,
                    adjusted_solution_min,
                    targets,
                    adjusted_reference,
                )
                outcome_dict = outcome_to_dict(outcome)

                solution_original = orient_objectives_from_minimize(problem, solution_min)
                adjusted_solution_original = orient_objectives_from_minimize(
                    problem, adjusted_solution_min
                )

                owen_values = sorted(
                    explanation["owen_values"],
                    key=lambda row: abs(float(row.get("owen_value", 0.0))),
                    reverse=True,
                )

                print(
                    f"{profile_name:11s} case={explanation['case_number']} "
                    f"outcome={outcome.outcome_type:19s} "
                    f"targets={','.join(targets)}"
                )

                row = {
                    "problem": args.problem,
                    "target_set": target_label,
                    "profile": profile_name,
                    "case_number": explanation["case_number"],
                    "case_name": explanation["case_name"],
                    "reference_point_status": explanation["reference_point_status"],
                    "suggestion": explanation["suggestion"],
                    "explanation": explanation["explanation"],
                    "actionable_targets": json_text(actionable_targets),
                    "selected_rivals": json_text(selected_rivals),
                    "strongest_supporter": json_text(
                        explanation["strongest_supporter_members"]
                    ),
                    "strongest_rival": json_text(explanation["strongest_rival_members"]),
                    "counterfactual_outcome": outcome.outcome_type,
                    "joint_target_improvement": outcome_dict["joint_target_improvement"],
                    "improved_targets": json_text(outcome_dict["improved_targets"]),
                    "worsened_targets": json_text(outcome_dict["worsened_targets"]),
                    "unchanged_targets": json_text(outcome_dict["unchanged_targets"]),
                    "target_changes": json_text(outcome_dict["changes"]),
                    "owen_values": json_text(owen_values),
                    "coalition_contributions": json_text(
                        explanation["coalition_contributions"]
                    ),
                }

                for index, symbol in enumerate(objective_symbols):
                    row[f"reference_{symbol}"] = float(reference_original[index])
                    row[f"solution_{symbol}"] = float(solution_original[index])
                    row[f"adjusted_reference_{symbol}"] = float(adjusted_reference[index])
                    row[f"adjusted_solution_{symbol}"] = float(
                        adjusted_solution_original[index]
                    )

                rows.append(row)

            except Exception as error:
                print(f"{profile_name:11s} FAILED: {error}")
                rows.append(
                    {
                        "problem": args.problem,
                        "target_set": target_label,
                        "profile": profile_name,
                        "case_number": None,
                        "case_name": None,
                        "reference_point_status": None,
                        "suggestion": None,
                        "explanation": None,
                        "actionable_targets": "[]",
                        "selected_rivals": "[]",
                        "strongest_supporter": "[]",
                        "strongest_rival": "[]",
                        "counterfactual_outcome": "failed",
                        "joint_target_improvement": False,
                        "improved_targets": "[]",
                        "worsened_targets": "[]",
                        "unchanged_targets": "[]",
                        "target_changes": "[]",
                        "owen_values": "[]",
                        "coalition_contributions": "[]",
                        "error": str(error),
                    }
                )

    results = pl.DataFrame(rows)
    output = args.output or Path("results") / (
        f"owen_explanation_evaluation_{args.problem}_{args.samples}samples_seed{args.seed}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(output)

    print("\nCOUNTERFACTUAL OUTCOME COUNTS")
    print("-" * 68)
    print(
        results.group_by("counterfactual_outcome")
        .len()
        .sort("len", descending=True)
    )

    successful = results.filter(pl.col("counterfactual_outcome") != "failed")
    if successful.height:
        print("\nOUTCOMES BY TARGET-SET SIZE / CASE")
        print("-" * 68)
        summary = (
            successful.with_columns(
                pl.col("target_set").str.count_matches(r"\+").add(1).alias("target_count")
            )
            .group_by(["target_count", "case_number", "counterfactual_outcome"])
            .len()
            .sort(["target_count", "case_number", "counterfactual_outcome"])
        )
        print(summary)

    print(f"\nSaved evaluation results to: {output}")


if __name__ == "__main__":
    main()
