"""Compare matched Owen hierarchy evaluation results.

This script compares two CSV files produced by
``evaluate_owen_explanations.py``. It is intended for controlled
comparisons between different Partition explainer hierarchies while
keeping the evaluated problem, target sets, and reference-point
profiles fixed.

The comparison focuses on three levels:

1. Owen attribution changes.
2. Generalized R-XIMO interpretation and recommendation changes.
3. Counterfactual behavior of the resulting recommendations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


MATCH_COLUMNS = [
    "problem",
    "target_set",
    "profile",
]


def parse_json(value: str | None):
    """Parse a JSON value stored in an evaluator CSV cell."""
    if value is None:
        return None

    return json.loads(value)


def parse_owen_values(value: str) -> dict[str, float]:
    """Return Owen values indexed by input symbol."""
    entries = parse_json(value)

    return {
        entry["input"]: float(entry["owen_value"])
        for entry in entries
    }


def parse_target_changes(value: str) -> dict[str, dict]:
    """Return target-change records indexed by objective symbol."""
    entries = parse_json(value)

    return {
        entry["symbol"]: entry
        for entry in entries
        if entry["is_target"]
    }


def sign_with_tolerance(
    value: float,
    *,
    tolerance: float = 1e-9,
) -> int:
    """Return the sign of a value using a numerical tolerance."""
    if value > tolerance:
        return 1

    if value < -tolerance:
        return -1

    return 0


def rank_owen_values(
    values: dict[str, float],
) -> list[str]:
    """Rank inputs from the largest to the smallest Owen value."""
    return [
        symbol
        for symbol, _ in sorted(
            values.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def compare_owen_values(
    first: dict[str, float],
    second: dict[str, float],
    *,
    sign_tolerance: float = 1e-9,
) -> dict:
    """Compare Owen values from two hierarchy configurations."""
    if set(first) != set(second):
        raise ValueError(
            "The compared rows contain different Owen input symbols."
        )

    symbols = sorted(first)

    differences = {
        symbol: second[symbol] - first[symbol]
        for symbol in symbols
    }

    sign_changes = [
        symbol
        for symbol in symbols
        if sign_with_tolerance(
            first[symbol],
            tolerance=sign_tolerance,
        )
        != sign_with_tolerance(
            second[symbol],
            tolerance=sign_tolerance,
        )
    ]

    first_ranking = rank_owen_values(first)
    second_ranking = rank_owen_values(second)

    return {
        "differences": differences,
        "sign_changes": sign_changes,
        "ranking_changed": first_ranking != second_ranking,
        "first_ranking": first_ranking,
        "second_ranking": second_ranking,
        "mean_absolute_change": float(
            np.mean(
                [
                    abs(differences[symbol])
                    for symbol in symbols
                ]
            )
        ),
        "max_absolute_change": float(
            max(
                abs(differences[symbol])
                for symbol in symbols
            )
        ),
    }


def compare_target_changes(
    first: dict[str, dict],
    second: dict[str, dict],
) -> dict[str, dict]:
    """Compare normalized counterfactual changes for target objectives."""
    symbols = sorted(set(first) | set(second))

    comparison = {}

    for symbol in symbols:
        first_record = first.get(symbol)
        second_record = second.get(symbol)

        first_change = (
            float(first_record["normalized_improvement"])
            if first_record is not None
            else np.nan
        )
        second_change = (
            float(second_record["normalized_improvement"])
            if second_record is not None
            else np.nan
        )

        comparison[symbol] = {
            "first": first_change,
            "second": second_change,
            "difference": second_change - first_change,
        }

    return comparison


def load_results(path: Path) -> pl.DataFrame:
    """Load one Owen evaluation result file."""
    data = pl.read_csv(path)

    missing = [
        column
        for column in MATCH_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{path} is missing matching columns: "
            + ", ".join(missing)
        )

    return data


def build_comparison(
    first: pl.DataFrame,
    second: pl.DataFrame,
    *,
    first_label: str,
    second_label: str,
    sign_tolerance: float,
) -> pl.DataFrame:
    """Build a matched scenario comparison table."""
    first_rows = {
        tuple(row[column] for column in MATCH_COLUMNS): row
        for row in first.to_dicts()
    }

    second_rows = {
        tuple(row[column] for column in MATCH_COLUMNS): row
        for row in second.to_dicts()
    }

    first_keys = set(first_rows)
    second_keys = set(second_rows)

    if first_keys != second_keys:
        only_first = sorted(first_keys - second_keys)
        only_second = sorted(second_keys - first_keys)

        raise ValueError(
            "Evaluation files do not contain the same scenarios.\n"
            f"Only in {first_label}: {only_first}\n"
            f"Only in {second_label}: {only_second}"
        )

    records = []

    for key in sorted(first_keys):
        first_row = first_rows[key]
        second_row = second_rows[key]

        first_owen = parse_owen_values(
            first_row["owen_values"]
        )
        second_owen = parse_owen_values(
            second_row["owen_values"]
        )

        owen_comparison = compare_owen_values(
            first_owen,
            second_owen,
            sign_tolerance=sign_tolerance,
        )

        first_targets = parse_target_changes(
            first_row["target_changes"]
        )
        second_targets = parse_target_changes(
            second_row["target_changes"]
        )

        target_comparison = compare_target_changes(
            first_targets,
            second_targets,
        )

        record = {
            "problem": first_row["problem"],
            "target_set": first_row["target_set"],
            "profile": first_row["profile"],
            "target_count": len(
                first_row["target_set"].split("+")
            ),

            f"{first_label}_case": first_row["case_number"],
            f"{second_label}_case": second_row["case_number"],
            "case_changed": (
                first_row["case_number"]
                != second_row["case_number"]
            ),

            f"{first_label}_selected_rivals":
                first_row["selected_rivals"],
            f"{second_label}_selected_rivals":
                second_row["selected_rivals"],
            "selected_rivals_changed": (
                first_row["selected_rivals"]
                != second_row["selected_rivals"]
            ),

            f"{first_label}_strongest_supporter":
                first_row["strongest_supporter"],
            f"{second_label}_strongest_supporter":
                second_row["strongest_supporter"],
            "strongest_supporter_changed": (
                first_row["strongest_supporter"]
                != second_row["strongest_supporter"]
            ),

            f"{first_label}_strongest_rival":
                first_row["strongest_rival"],
            f"{second_label}_strongest_rival":
                second_row["strongest_rival"],
            "strongest_rival_changed": (
                first_row["strongest_rival"]
                != second_row["strongest_rival"]
            ),

            f"{first_label}_suggestion":
                first_row["suggestion"],
            f"{second_label}_suggestion":
                second_row["suggestion"],
            "suggestion_changed": (
                first_row["suggestion"]
                != second_row["suggestion"]
            ),

            "owen_sign_changed": bool(
                owen_comparison["sign_changes"]
            ),
            "owen_sign_changed_inputs": json.dumps(
                owen_comparison["sign_changes"]
            ),
            "owen_ranking_changed":
                owen_comparison["ranking_changed"],
            f"{first_label}_owen_ranking": json.dumps(
                owen_comparison["first_ranking"]
            ),
            f"{second_label}_owen_ranking": json.dumps(
                owen_comparison["second_ranking"]
            ),
            "mean_absolute_owen_change":
                owen_comparison["mean_absolute_change"],
            "max_absolute_owen_change":
                owen_comparison["max_absolute_change"],

            f"{first_label}_counterfactual_outcome":
                first_row["counterfactual_outcome"],
            f"{second_label}_counterfactual_outcome":
                second_row["counterfactual_outcome"],
            "counterfactual_outcome_changed": (
                first_row["counterfactual_outcome"]
                != second_row["counterfactual_outcome"]
            ),

            f"{first_label}_target_utility_change":
                first_row["target_utility_change"],
            f"{second_label}_target_utility_change":
                second_row["target_utility_change"],
            "target_utility_change_difference": (
                float(second_row["target_utility_change"])
                - float(first_row["target_utility_change"])
            ),
        }

        for symbol in sorted(first_owen):
            record[f"{first_label}_owen_{symbol}"] = (
                first_owen[symbol]
            )
            record[f"{second_label}_owen_{symbol}"] = (
                second_owen[symbol]
            )
            record[f"delta_owen_{symbol}"] = (
                owen_comparison["differences"][symbol]
            )

        for symbol, changes in target_comparison.items():
            record[
                f"{first_label}_target_change_{symbol}"
            ] = changes["first"]

            record[
                f"{second_label}_target_change_{symbol}"
            ] = changes["second"]

            record[
                f"delta_target_change_{symbol}"
            ] = changes["difference"]

        records.append(record)

    return pl.DataFrame(records)


def print_summary(
    comparison: pl.DataFrame,
    *,
    first_label: str,
    second_label: str,
) -> None:
    """Print a concise summary of hierarchy differences."""
    total = comparison.height

    print()
    print("OWEN HIERARCHY COMPARISON")
    print("=" * 68)
    print(f"First hierarchy:      {first_label}")
    print(f"Second hierarchy:     {second_label}")
    print(f"Matched scenarios:    {total}")
    print()

    metrics = [
        ("Owen sign changed", "owen_sign_changed"),
        ("Owen ranking changed", "owen_ranking_changed"),
        (
            "Strongest supporter changed",
            "strongest_supporter_changed",
        ),
        (
            "Strongest rival changed",
            "strongest_rival_changed",
        ),
        ("R-XIMO case changed", "case_changed"),
        (
            "Selected rivals changed",
            "selected_rivals_changed",
        ),
        ("Suggestion changed", "suggestion_changed"),
        (
            "Counterfactual outcome changed",
            "counterfactual_outcome_changed",
        ),
    ]

    for label, column in metrics:
        count = comparison[column].sum()
        print(
            f"{label:<34} "
            f"{count:>3} / {total}"
        )

    print()
    print("OWEN VALUE DIFFERENCES")
    print("-" * 68)

    mean_change = comparison[
        "mean_absolute_owen_change"
    ].mean()

    max_change = comparison[
        "max_absolute_owen_change"
    ].max()

    print(
        f"Mean scenario-level absolute change: "
        f"{mean_change:.6f}"
    )
    print(
        f"Maximum absolute change:             "
        f"{max_change:.6f}"
    )

    changed = comparison.filter(
        pl.col("suggestion_changed")
    )

    print()
    print("SCENARIOS WITH CHANGED SUGGESTIONS")
    print("-" * 68)

    if changed.height == 0:
        print("None.")
        return

    for row in changed.to_dicts():
        print(
            f"{row['target_set']:<12} "
            f"{row['profile']:<12} "
            f"case {row[f'{first_label}_case']} -> "
            f"{row[f'{second_label}_case']}"
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
            "  Utility change: "
            f"{format_value(float(row[f'{first_label}_target_utility_change']))}"
            " -> "
            f"{format_value(float(row[f'{second_label}_target_utility_change']))}"
        )

        target_symbols = row["target_set"].split("+")

        for symbol in target_symbols:
            first_change = row.get(
                f"{first_label}_target_change_{symbol}"
            )
            second_change = row.get(
                f"{second_label}_target_change_{symbol}"
            )

            if (
                first_change is not None
                and second_change is not None
            ):
                print(
                    f"  {symbol}: "
                    f"{format_value(float(first_change))}"
                    " -> "
                    f"{format_value(float(second_change))}"
                )

        print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched Owen hierarchy evaluation CSV files."
        )
    )

    parser.add_argument(
        "first",
        type=Path,
        help="First evaluation CSV.",
    )
    parser.add_argument(
        "second",
        type=Path,
        help="Second evaluation CSV.",
    )
    parser.add_argument(
        "--first-label",
        default="correlation",
        help="Short label for the first hierarchy.",
    )
    parser.add_argument(
        "--second-label",
        default="moo",
        help="Short label for the second hierarchy.",
    )
    parser.add_argument(
        "--sign-tolerance",
        type=float,
        default=1e-9,
        help=(
            "Tolerance used when determining Owen-value signs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )

    return parser.parse_args()


def format_value(value: float) -> str:
    """Format a numeric result without displaying negative zero."""
    if round(value, 6) == 0:
        value = 0.0

    return f"{value:+.6f}"


def main() -> None:
    """Run the hierarchy comparison."""
    args = parse_args()

    first = load_results(args.first)
    second = load_results(args.second)

    comparison = build_comparison(
        first,
        second,
        first_label=args.first_label,
        second_label=args.second_label,
        sign_tolerance=args.sign_tolerance,
    )

    print_summary(
        comparison,
        first_label=args.first_label,
        second_label=args.second_label,
    )

    output = args.output

    if output is None:
        output = Path("results") / (
            "owen_hierarchy_comparison.csv"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.write_csv(output)

    print(f"Saved comparison results to: {output}")


if __name__ == "__main__":
    main()
