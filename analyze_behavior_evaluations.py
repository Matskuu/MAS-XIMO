"""Analyze behavioral results from Owen explanation evaluation CSV files.

The script is problem-independent and accepts one or more CSV files produced by
``evaluate_owen_explanations.py``. It summarizes scenario coverage, generalized
R-XIMO cases, reference-point statuses, counterfactual outcomes, target-utility
changes, external adjustment patterns, and target-set effects.

When multiple evaluation seeds are supplied, the script also measures cross-seed
stability of attribution, generalized R-XIMO reasoning, recommendations, and
counterfactual behavior for matched profile/target scenarios.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median, pstdev

import polars as pl


REQUIRED_COLUMNS = {
    "problem",
    "target_set",
    "profile",
    "case_number",
    "case_name",
    "reference_point_status",
    "suggestion",
    "selected_rivals",
    "counterfactual_outcome",
    "joint_target_improvement",
    "target_utility_change",
    "target_utility_improved",
}


SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)
OWEN_EPSILON = 1e-9


def seed_label(source_file: object) -> str:
    """Return a stable seed label inferred from an evaluation filename."""
    text = str(source_file)
    match = SEED_PATTERN.search(Path(text).name)
    return f"seed{match.group(1)}" if match else Path(text).stem


def parse_owen_values(value: object) -> dict[str, float]:
    """Parse the evaluator's serialized Owen-value records."""
    if value is None:
        return {}

    if isinstance(value, list):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return {}
        parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected Owen-value list, got: {value!r}")

    values: dict[str, float] = {}
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"Expected Owen-value record, got: {item!r}")

        symbol = str(item["input"])
        values[symbol] = float(item["owen_value"])

    return values


def owen_signs(value: object, tolerance: float = 1e-9) -> tuple:
    """Return the Owen sign pattern in deterministic objective order."""
    values = parse_owen_values(value)

    def sign(number: float) -> int:
        if number > tolerance:
            return 1
        if number < -tolerance:
            return -1
        return 0

    return tuple(
        (symbol, sign(number))
        for symbol, number in sorted(values.items())
    )


def owen_ranking(value: object) -> tuple[str, ...]:
    """Return inputs ranked by descending absolute Owen value."""
    values = parse_owen_values(value)

    return tuple(
        symbol
        for symbol, _ in sorted(
            values.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )
    )


def canonical_external_selection(value: object) -> tuple[str, ...]:
    """Return selected external effects in order-independent canonical form."""
    return tuple(sorted(parse_json_list(value)))


def stable_count(groups: list[list[dict]], extractor) -> int:
    """Count matched scenarios whose extracted value agrees across all seeds."""
    return sum(
        1
        for group in groups
        if len({extractor(row) for row in group}) == 1
    )


def print_stability_metric(
    label: str,
    groups: list[list[dict]],
    extractor,
) -> None:
    """Print all/single/multi cross-seed stability for one metric."""
    stable = stable_count(groups, extractor)
    single = [g for g in groups if target_kind(g[0]["target_set"]) == "single"]
    multi = [g for g in groups if target_kind(g[0]["target_set"]) == "multi"]
    parts = [f"all={percentage(stable, len(groups))}"]
    if single:
        parts.append(f"single={percentage(stable_count(single, extractor), len(single))}")
    if multi:
        parts.append(f"multi={percentage(stable_count(multi, extractor), len(multi))}")
    print(f"{label:<30} " + "; ".join(parts))


def analyze_cross_seed_stability(problem: str, frame: pl.DataFrame) -> None:
    """Analyze matched explanation scenarios across multiple background seeds."""
    rows = frame.to_dicts()
    for row in rows:
        row["_seed"] = seed_label(row["source_file"])

    seeds = sorted({row["_seed"] for row in rows})
    if len(seeds) < 2:
        return

    by_scenario: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        key = (str(row["profile"]), canonical_target_set(row["target_set"]))
        seed_rows = by_scenario.setdefault(key, {})
        if row["_seed"] in seed_rows:
            raise ValueError(
                f"Duplicate {problem} scenario {key} for {row['_seed']}."
            )
        seed_rows[row["_seed"]] = row

    complete = {
        key: seed_rows
        for key, seed_rows in by_scenario.items()
        if set(seed_rows) == set(seeds)
    }
    incomplete = len(by_scenario) - len(complete)
    groups = [[seed_rows[seed] for seed in seeds] for seed_rows in complete.values()]

    print("=" * 76)
    print(f"CROSS-SEED STABILITY: {problem}")
    print("=" * 76)
    print(f"Seeds: {', '.join(seeds)}")
    print(f"Matched scenarios: {len(groups)}")
    if incomplete:
        print(f"Incomplete scenarios excluded: {incomplete}")
    print()

    columns = set(frame.columns)
    metrics = []
    if "owen_values" in columns:
        metrics.extend([
            ("Owen signs", lambda row: owen_signs(row["owen_values"])),
            ("Owen ranking", lambda row: owen_ranking(row["owen_values"])),
        ])
    if "strongest_supporter" in columns:
        metrics.append(("Strongest supporter", lambda row: str(row["strongest_supporter"])))
    if "strongest_rival" in columns:
        metrics.append(("Strongest rival", lambda row: str(row["strongest_rival"])))
    metrics.extend([
        ("R-XIMO case", lambda row: int(row["case_number"])),
        ("Selected external effects", lambda row: canonical_external_selection(row["selected_rivals"])),
        ("DM-facing suggestion", lambda row: str(row["suggestion"])),
        ("Counterfactual outcome", lambda row: str(row["counterfactual_outcome"])),
        ("Joint target improvement", lambda row: bool(row["joint_target_improvement"])),
    ])
    if "coalition_outcome" in columns:
        metrics.append(("Coalition outcome", lambda row: str(row["coalition_outcome"])))

    print("Stability across all supplied seeds")
    print("-----------------------------------")
    for label, extractor in metrics:
        print_stability_metric(label, groups, extractor)
    print()

    print("Cross-seed target-utility variability")
    print("-------------------------------------")

    def utility_variability_summary(selected_groups: list[list[dict]]) -> tuple[list[float], list[float]]:
        ranges = []
        deviations = []
        for group in selected_groups:
            values = [float(row["target_utility_change"]) for row in group]
            ranges.append(max(values) - min(values))
            deviations.append(pstdev(values))
        return ranges, deviations

    for label, selected_groups in (
        ("all", groups),
        ("single", [g for g in groups if target_kind(g[0]["target_set"]) == "single"]),
        ("multi", [g for g in groups if target_kind(g[0]["target_set"]) == "multi"]),
    ):
        if not selected_groups:
            continue
        ranges, deviations = utility_variability_summary(selected_groups)
        print(
            f"{label:<8} range: "
            f"mean={mean(ranges):.6f}, median={median(ranges):.6f}, "
            f"max={max(ranges):.6f}"
        )
        print(
            f"{'':8} std:   "
            f"mean={mean(deviations):.6f}, median={median(deviations):.6f}, "
            f"max={max(deviations):.6f}"
        )

    if groups:
        worst_group = max(
            groups,
            key=lambda group: (
                max(float(row["target_utility_change"]) for row in group)
                - min(float(row["target_utility_change"]) for row in group)
            ),
        )
        worst_values = [float(row["target_utility_change"]) for row in worst_group]
        worst_profile = str(worst_group[0]["profile"])
        worst_targets = canonical_target_set(worst_group[0]["target_set"])
        seed_values = ", ".join(
            f"{row['_seed']}={float(row['target_utility_change']):+.6f}"
            for row in worst_group
        )
        print(
            "Largest range: "
            f"profile={worst_profile}, targets={worst_targets}, "
            f"range={max(worst_values) - min(worst_values):.6f}"
        )
        print(f"  {seed_values}")
    print()

    print("Pairwise seed comparisons")
    print("-------------------------")
    for seed_a, seed_b in combinations(seeds, 2):
        paired = [
            [seed_rows[seed_a], seed_rows[seed_b]]
            for seed_rows in complete.values()
        ]
        print(f"{seed_a} vs {seed_b}")
        for label, extractor in metrics:
            changed = len(paired) - stable_count(paired, extractor)
            print(f"  {label:<28} changed={percentage(changed, len(paired))}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Owen explanation behavior across evaluation CSVs."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Evaluation CSV files to analyze.",
    )
    return parser.parse_args()


def parse_json_list(value: object) -> list[str]:
    """Parse a JSON-serialized list, returning an empty list for null values."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got: {value!r}")
    return [str(item) for item in parsed]


def target_symbols(target_set: object) -> list[str]:
    """Parse a target-set field such as 'f_1' or 'f_1,f_3'."""
    text = str(target_set).strip()
    if not text:
        return []
    # Evaluation files may serialize multi-target sets with ``+`` or commas.
    # Supporting whitespace as a fallback keeps the analyzer tolerant of older
    # experimental files.
    if "+" in text:
        return [item.strip() for item in text.split("+") if item.strip()]
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [item.strip() for item in text.split() if item.strip()]


def objective_symbols(frame: pl.DataFrame) -> list[str]:
    """Infer objective symbols from reference_<symbol> columns."""
    symbols = []
    for column in frame.columns:
        if not column.startswith("reference_"):
            continue
        symbol = column.removeprefix("reference_")
        # Objective reference-point columns have names such as reference_f_1.
        # Exclude derived fields such as reference_point_status and
        # reference_point_changes rather than treating them as objectives.
        if symbol.startswith("f_") and symbol[2:].isdigit():
            symbols.append(symbol)
    return sorted(symbols, key=lambda symbol: int(symbol.split("_")[-1]))


def expected_target_sets(symbols: list[str]) -> set[str]:
    """Return canonical singleton and pair target sets."""
    sets = {symbol for symbol in symbols}
    sets.update(
        ",".join(pair)
        for pair in combinations(symbols, 2)
    )
    return sets


def canonical_target_set(value: object) -> str:
    return ",".join(target_symbols(value))


def target_kind(value: object) -> str:
    size = len(target_symbols(value))
    if size == 1:
        return "single"
    if size >= 2:
        return "multi"
    return "unknown"


def percentage(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / total:.1f}%)" if total else "0/0"


def print_counter(title: str, counter: Counter, total: int) -> None:
    print(title)
    print("-" * len(title))
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"{str(key):<32} {percentage(count, total)}")
    print()


def utility_summary(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={mean(values):+.6f}, median={median(values):+.6f}, "
        f"min={min(values):+.6f}, max={max(values):+.6f}"
    )


def analyze_problem(problem: str, frame: pl.DataFrame) -> None:
    rows = frame.to_dicts()
    total = len(rows)
    symbols = objective_symbols(frame)
    profiles = sorted({str(row["profile"]) for row in rows})

    print("=" * 76)
    print(f"PROBLEM: {problem}")
    print("=" * 76)
    print(f"Scenarios: {total}")
    print(f"Objectives ({len(symbols)}): {', '.join(symbols)}")
    print(f"Profiles ({len(profiles)}): {', '.join(profiles)}")

    observed_pairs = [
        (str(row["profile"]), canonical_target_set(row["target_set"]))
        for row in rows
    ]
    pair_counts = Counter(observed_pairs)
    duplicates = [pair for pair, count in pair_counts.items() if count > 1]

    expected_targets = expected_target_sets(symbols)
    observed_targets = {canonical_target_set(row["target_set"]) for row in rows}
    missing_targets = sorted(expected_targets - observed_targets)
    extra_targets = sorted(observed_targets - expected_targets)
    expected_scenarios = len(profiles) * len(expected_targets)

    missing_scenarios = sorted(
        (profile, targets)
        for profile in profiles
        for targets in expected_targets
        if (profile, targets) not in pair_counts
    )

    print(
        "Coverage: "
        f"{total}/{expected_scenarios} expected singleton/pair scenarios"
    )
    if missing_targets:
        print(f"Missing target sets: {', '.join(missing_targets)}")
    if extra_targets:
        print(f"Extra target sets: {', '.join(extra_targets)}")
    if missing_scenarios:
        print(f"Missing profile/target scenarios: {len(missing_scenarios)}")
    if duplicates:
        print(f"Duplicate profile/target scenarios: {len(duplicates)}")
    if not (missing_targets or extra_targets or missing_scenarios or duplicates):
        print("Coverage check: OK")
    print()

    kinds = Counter(target_kind(row["target_set"]) for row in rows)
    print_counter("Target configuration", kinds, total)

    case_counter = Counter(
        f"Case {int(row['case_number'])}: {row['case_name']}" for row in rows
    )
    print_counter("Generalized R-XIMO cases", case_counter, total)

    for kind in ("single", "multi"):
        subset = [row for row in rows if target_kind(row["target_set"]) == kind]
        if not subset:
            continue
        cases = Counter(
            f"Case {int(row['case_number'])}" for row in subset
        )
        print_counter(
            f"R-XIMO cases — {kind} target",
            cases,
            len(subset),
        )

    statuses = Counter(str(row["reference_point_status"]) for row in rows)
    print_counter("Reference-point status", statuses, total)

    print("Reference-point status by profile")
    print("---------------------------------")
    for profile in profiles:
        subset = [row for row in rows if str(row["profile"]) == profile]
        counts = Counter(str(row["reference_point_status"]) for row in subset)
        text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"{profile:<16} {text}")
    print()

    outcomes = Counter(str(row["counterfactual_outcome"]) for row in rows)
    print_counter("Counterfactual outcomes", outcomes, total)

    for kind in ("single", "multi"):
        subset = [row for row in rows if target_kind(row["target_set"]) == kind]
        if not subset:
            continue
        counts = Counter(str(row["counterfactual_outcome"]) for row in subset)
        print_counter(
            f"Counterfactual outcomes — {kind} target",
            counts,
            len(subset),
        )

    utility_values = [float(row["target_utility_change"]) for row in rows]
    print("Target-utility change")
    print("---------------------")
    print(f"all:    {utility_summary(utility_values)}")
    for kind in ("single", "multi"):
        values = [
            float(row["target_utility_change"])
            for row in rows
            if target_kind(row["target_set"]) == kind
        ]
        print(f"{kind + ':':<8}{utility_summary(values)}")
    improved = sum(bool(row["target_utility_improved"]) for row in rows)
    print(f"utility improved: {percentage(improved, total)}")
    print()

    joint = sum(bool(row["joint_target_improvement"]) for row in rows)
    print("Joint target improvement")
    print("------------------------")
    print(percentage(joint, total))
    multi_rows = [row for row in rows if target_kind(row["target_set"]) == "multi"]
    if multi_rows:
        multi_joint = sum(bool(row["joint_target_improvement"]) for row in multi_rows)
        print(f"multi-target only: {percentage(multi_joint, len(multi_rows))}")
    print()

    external_counter: Counter[str] = Counter()
    external_count_distribution: Counter[int] = Counter()
    for row in rows:
        selected = parse_json_list(row["selected_rivals"])
        external_count_distribution[len(selected)] += 1
        external_counter.update(selected)

    print_counter(
        "Number of selected external objectives/effects",
        external_count_distribution,
        total,
    )
    if external_counter:
        print_counter(
            "Selected external objectives/effects",
            external_counter,
            sum(external_counter.values()),
        )

    print("Target-set effects within the same profile")
    print("------------------------------------------")
    for profile in profiles:
        subset = [row for row in rows if str(row["profile"]) == profile]
        suggestions = {str(row["suggestion"]) for row in subset}
        cases = {int(row["case_number"]) for row in subset}
        external_sets = {
            tuple(parse_json_list(row["selected_rivals"]))
            for row in subset
        }
        print(
            f"{profile:<16} "
            f"target sets={len(subset):2d}, "
            f"distinct cases={len(cases)}, "
            f"distinct external selections={len(external_sets)}, "
            f"distinct suggestions={len(suggestions)}"
        )
    print()


def main() -> None:
    args = parse_args()

    frames = []
    for path in args.files:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pl.read_csv(path)
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        frame = frame.with_columns(pl.lit(str(path)).alias("source_file"))
        frames.append(frame)

    combined = pl.concat(frames, how="diagonal_relaxed")

    print("Owen explanation behavioral analysis")
    print("=" * 76)
    print(f"Files: {len(args.files)}")
    print(f"Rows: {combined.height}")
    print()

    problems = sorted(combined.get_column("problem").unique().to_list())
    for problem in problems:
        problem_frame = combined.filter(pl.col("problem") == problem)
        source_files = sorted(problem_frame.get_column("source_file").unique().to_list())
        if len(source_files) == 1:
            analyze_problem(str(problem), problem_frame)
        else:
            for source_file in source_files:
                print(f"DATASET: {Path(str(source_file)).name}")
                analyze_problem(
                    str(problem),
                    problem_frame.filter(pl.col("source_file") == source_file),
                )
            analyze_cross_seed_stability(str(problem), problem_frame)

    if len(problems) > 1:
        print("=" * 76)
        print("CROSS-PROBLEM OVERVIEW")
        print("=" * 76)
        for problem in problems:
            subset = combined.filter(pl.col("problem") == problem)
            rows = subset.to_dicts()
            outcomes = Counter(str(row["counterfactual_outcome"]) for row in rows)
            cases = sorted({int(row["case_number"]) for row in rows})
            utility = [float(row["target_utility_change"]) for row in rows]
            print(
                f"{problem}: scenarios={len(rows)}, "
                f"cases={cases}, "
                f"outcomes={dict(outcomes)}, "
                f"mean Δutility={mean(utility):+.6f}"
            )


if __name__ == "__main__":
    main()
