"""Summarize Owen hierarchy comparisons across multiple seeds.

This script combines CSV files produced by
``compare_hierarchy_evaluations.py`` and summarizes how changing the
Partition explainer hierarchy affects Owen attribution, generalized
R-XIMO reasoning, recommendations, and counterfactual behavior across
multiple independently generated background datasets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl


BOOLEAN_METRICS = [
    "owen_sign_changed",
    "owen_ranking_changed",
    "strongest_supporter_changed",
    "strongest_rival_changed",
    "case_changed",
    "selected_rivals_changed",
    "suggestion_changed",
    "counterfactual_outcome_changed",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Owen hierarchy comparisons across seeds."
        )
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "Comparison CSV files produced by "
            "compare_hierarchy_evaluations.py."
        ),
    )

    parser.add_argument(
        "--first-label",
        default="correlation",
        help="Label used for the first hierarchy.",
    )

    parser.add_argument(
        "--second-label",
        default="moo",
        help="Label used for the second hierarchy.",
    )

    parser.add_argument(
        "--utility-tolerance",
        type=float,
        default=1e-9,
        help=(
            "Numerical tolerance used only when comparing target "
            "utility changes between hierarchies."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results")
        / "owen_hierarchy_summary_all_seeds.csv",
        help="Output CSV containing all combined comparisons.",
    )

    return parser.parse_args()


def extract_seed(path: Path) -> int:
    """Extract the seed number from a comparison filename."""
    match = re.search(r"seed(\d+)", path.stem)

    if match is None:
        raise ValueError(
            f"Could not determine seed from filename: {path}"
        )

    return int(match.group(1))


def count_relaxed_objectives(
    value: str | None,
) -> int:
    """Count selected external objectives from the stored CSV value."""
    if value is None:
        return 0

    value = str(value).strip()

    if value in {"", "[]", "null", "None"}:
        return 0

    # selected_rivals is stored as a serialized list.
    return len(
        re.findall(r"r_f_\d+|f_\d+", value)
    )


def load_comparisons(
    paths: list[Path],
) -> pl.DataFrame:
    """Load and combine hierarchy comparison CSV files."""
    frames = []

    for path in paths:
        data = pl.read_csv(path)
        seed = extract_seed(path)

        data = data.with_columns(
            pl.lit(seed).alias("seed")
        )

        frames.append(data)

    if not frames:
        raise ValueError("No comparison files were provided.")

    return pl.concat(
        frames,
        how="diagonal_relaxed",
    )


def add_analysis_columns(
    data: pl.DataFrame,
    *,
    first_label: str,
    second_label: str,
    utility_tolerance: float,
) -> pl.DataFrame:
    """Add aggregate-analysis columns."""
    first_utility = (
        f"{first_label}_target_utility_change"
    )
    second_utility = (
        f"{second_label}_target_utility_change"
    )

    first_rivals = f"{first_label}_selected_rivals"
    second_rivals = f"{second_label}_selected_rivals"

    data = data.with_columns(
        (
            pl.col(second_utility)
            - pl.col(first_utility)
        ).alias("utility_difference")
    )

    data = data.with_columns(
        pl.when(
            pl.col("utility_difference")
            > utility_tolerance
        )
        .then(pl.lit("second_higher"))
        .when(
            pl.col("utility_difference")
            < -utility_tolerance
        )
        .then(pl.lit("first_higher"))
        .otherwise(pl.lit("equivalent"))
        .alias("utility_comparison")
    )

    first_counts = [
        count_relaxed_objectives(value)
        for value in data[first_rivals].to_list()
    ]

    second_counts = [
        count_relaxed_objectives(value)
        for value in data[second_rivals].to_list()
    ]

    data = data.with_columns(
        pl.Series(
            f"{first_label}_relaxation_count",
            first_counts,
        ),
        pl.Series(
            f"{second_label}_relaxation_count",
            second_counts,
        ),
    )

    data = data.with_columns(
        (
            pl.col(
                f"{second_label}_relaxation_count"
            )
            - pl.col(
                f"{first_label}_relaxation_count"
            )
        ).alias("relaxation_count_difference")
    )

    data = data.with_columns(
        pl.when(
            pl.col("relaxation_count_difference") < 0
        )
        .then(pl.lit("second_fewer"))
        .when(
            pl.col("relaxation_count_difference") > 0
        )
        .then(pl.lit("second_more"))
        .otherwise(pl.lit("same"))
        .alias("relaxation_count_comparison")
    )

    return data


def print_metric_summary(
    data: pl.DataFrame,
    *,
    title: str,
) -> None:
    """Print counts for hierarchy-sensitive metrics."""
    print()
    print(title)
    print("-" * 72)

    total = data.height

    for metric in BOOLEAN_METRICS:
        count = int(data[metric].sum())

        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        label = metric.replace("_", " ")

        print(
            f"{label:<36}"
            f"{count:>4} / {total:<4}"
            f"({percentage:>6.2f}%)"
        )


def print_target_size_summary(
    data: pl.DataFrame,
) -> None:
    """Print separate summaries for single- and multi-target cases."""
    for target_count, label in [
        (1, "SINGLE-TARGET"),
        (2, "MULTI-TARGET"),
    ]:
        subset = data.filter(
            pl.col("target_count") == target_count
        )

        print_metric_summary(
            subset,
            title=f"{label} COMPARISONS",
        )


def print_seed_summary(
    data: pl.DataFrame,
) -> None:
    """Print recommendation effects separately for each seed."""
    print()
    print("RESULTS BY SEED")
    print("-" * 72)

    seeds = sorted(
        data["seed"].unique().to_list()
    )

    for seed in seeds:
        subset = data.filter(
            pl.col("seed") == seed
        )

        suggestion_changes = int(
            subset["suggestion_changed"].sum()
        )

        case_changes = int(
            subset["case_changed"].sum()
        )

        outcome_changes = int(
            subset[
                "counterfactual_outcome_changed"
            ].sum()
        )

        print(
            f"Seed {seed}: "
            f"suggestions={suggestion_changes}, "
            f"cases={case_changes}, "
            f"counterfactual outcomes={outcome_changes}"
        )


def print_recurrent_scenarios(
    data: pl.DataFrame,
) -> None:
    """Print scenarios whose suggestions changed in multiple seeds."""
    changed = data.filter(
        pl.col("suggestion_changed")
    )

    recurrence = (
        changed.group_by(
            ["target_set", "profile"]
        )
        .agg(
            pl.len().alias("changed_seed_count"),
            pl.col("seed")
            .sort()
            .alias("seeds"),
        )
        .sort(
            "changed_seed_count",
            descending=True,
        )
    )

    print()
    print("RECURRENT SUGGESTION CHANGES")
    print("-" * 72)

    recurrent = recurrence.filter(
        pl.col("changed_seed_count") > 1
    )

    if recurrent.height == 0:
        print(
            "No scenario changed its suggestion "
            "in more than one seed."
        )
        return

    for row in recurrent.to_dicts():
        seeds = ", ".join(
            str(seed)
            for seed in row["seeds"]
        )

        print(
            f"{row['target_set']:<12}"
            f"{row['profile']:<14}"
            f"{row['changed_seed_count']} seeds "
            f"({seeds})"
        )


def print_changed_suggestion_analysis(
    data: pl.DataFrame,
    *,
    first_label: str,
    second_label: str,
) -> None:
    """Summarize consequences when the suggestion changes."""
    changed = data.filter(
        pl.col("suggestion_changed")
    )

    print()
    print("CHANGED-SUGGESTION CONSEQUENCES")
    print("-" * 72)
    print(
        f"Changed suggestions: {changed.height}"
    )

    if changed.height == 0:
        return

    utility_counts = (
        changed.group_by("utility_comparison")
        .len()
        .sort("utility_comparison")
    )

    print()
    print("Target utility:")
    for row in utility_counts.to_dicts():
        label = row["utility_comparison"]

        if label == "second_higher":
            description = (
                f"{second_label} produced higher utility"
            )
        elif label == "first_higher":
            description = (
                f"{first_label} produced higher utility"
            )
        else:
            description = "equivalent utility"

        print(
            f"  {description:<40}"
            f"{row['len']}"
        )

    relaxation_counts = (
        changed.group_by(
            "relaxation_count_comparison"
        )
        .len()
        .sort("relaxation_count_comparison")
    )

    print()
    print("Number of relaxed objectives:")
    for row in relaxation_counts.to_dicts():
        label = row[
            "relaxation_count_comparison"
        ]

        if label == "second_fewer":
            description = (
                f"{second_label} used fewer"
            )
        elif label == "second_more":
            description = (
                f"{second_label} used more"
            )
        else:
            description = "same number"

        print(
            f"  {description:<40}"
            f"{row['len']}"
        )

    outcome_changed = int(
        changed[
            "counterfactual_outcome_changed"
        ].sum()
    )

    print()
    print(
        "Changed counterfactual outcome category: "
        f"{outcome_changed} / {changed.height}"
    )


def print_changed_scenarios(
    data: pl.DataFrame,
    *,
    first_label: str,
    second_label: str,
) -> None:
    """Print every changed recommendation across seeds."""
    changed = (
        data.filter(
            pl.col("suggestion_changed")
        )
        .sort(
            ["seed", "target_set", "profile"]
        )
    )

    print()
    print("ALL CHANGED SUGGESTIONS")
    print("-" * 72)

    for row in changed.to_dicts():
        print(
            f"Seed {row['seed']} | "
            f"{row['target_set']} | "
            f"{row['profile']}"
        )

        print(
            f"  {first_label}: "
            f"{row[f'{first_label}_suggestion']}"
        )

        print(
            f"  {second_label}: "
            f"{row[f'{second_label}_suggestion']}"
        )

        print(
            "  Utility: "
            f"{float(row[f'{first_label}_target_utility_change']):+.6f}"
            " -> "
            f"{float(row[f'{second_label}_target_utility_change']):+.6f}"
        )

        print(
            "  Outcome: "
            f"{row[f'{first_label}_counterfactual_outcome']}"
            " -> "
            f"{row[f'{second_label}_counterfactual_outcome']}"
        )

        print(
            "  Relaxed objectives: "
            f"{row[f'{first_label}_relaxation_count']}"
            " -> "
            f"{row[f'{second_label}_relaxation_count']}"
        )

        print()


def main() -> None:
    """Run the aggregate hierarchy analysis."""
    args = parse_args()

    data = load_comparisons(args.inputs)

    data = add_analysis_columns(
        data,
        first_label=args.first_label,
        second_label=args.second_label,
        utility_tolerance=args.utility_tolerance,
    )

    print()
    print("AGGREGATE OWEN HIERARCHY ANALYSIS")
    print("=" * 72)
    print(
        f"Seeds:               "
        f"{sorted(data['seed'].unique().to_list())}"
    )
    print(
        f"Total comparisons:   {data.height}"
    )
    print(
        f"Single-target:       "
        f"{data.filter(pl.col('target_count') == 1).height}"
    )
    print(
        f"Multi-target:        "
        f"{data.filter(pl.col('target_count') == 2).height}"
    )

    print_metric_summary(
        data,
        title="ALL COMPARISONS",
    )

    print_target_size_summary(data)

    print_seed_summary(data)

    print_recurrent_scenarios(data)

    print_changed_suggestion_analysis(
        data,
        first_label=args.first_label,
        second_label=args.second_label,
    )

    print_changed_scenarios(
        data,
        first_label=args.first_label,
        second_label=args.second_label,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.write_csv(args.output)

    print(
        f"Saved aggregate results to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
