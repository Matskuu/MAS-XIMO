"""Analyze objective relationships for MOO-informed Owen hierarchies.

This script examines relationships among Pareto-optimal solution objective
values stored in an R-XIMO background dataset. It constructs candidate
distance matrices and hierarchical clusterings that may later be supplied to
SHAP's PartitionExplainer as problem-informed coalition structures.

The analysis is exploratory. Correlation is treated as an empirical measure
of objective relationship, not as a direct definition of objective conflict.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from problem_setup import (
    PROBLEM_CHOICES,
    default_background_path,
)


def get_solution_columns(data: pl.DataFrame) -> list[str]:
    """Return solution-objective columns in dataset order."""
    columns = [
        column
        for column in data.columns
        if column.startswith("s_f_")
    ]

    if not columns:
        raise ValueError(
            "Dataset does not contain solution columns with prefix 's_f_'."
        )

    return columns


def compute_pearson_correlation(
    solution_values: np.ndarray,
) -> np.ndarray:
    """Compute Pearson correlations among solution objectives."""
    return np.corrcoef(
        solution_values,
        rowvar=False,
    )


def compute_spearman_correlation(
    solution_values: np.ndarray,
) -> np.ndarray:
    """Compute Spearman correlations among solution objectives."""
    ranked = np.empty_like(
        solution_values,
        dtype=float,
    )

    for column_index in range(solution_values.shape[1]):
        values = solution_values[:, column_index]

        ranked[:, column_index] = (
            pl.Series(values)
            .rank(method="average")
            .to_numpy()
        )

    return np.corrcoef(
        ranked,
        rowvar=False,
    )


def relationship_distance(
    correlation: np.ndarray,
) -> np.ndarray:
    """Convert correlations to unsigned relationship distances.

    Strong positive and negative relationships are both considered close.

    d_ij = 1 - |rho_ij|
    """
    distance = 1.0 - np.abs(correlation)
    distance = np.clip(distance, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def conflict_distance(
    correlation: np.ndarray,
) -> np.ndarray:
    """Convert correlations to conflict-focused distances.

    Strong negative correlations are considered close.

    d_ij = (1 + rho_ij) / 2

    The diagonal is set to zero because self-distance is zero.
    """
    distance = (1.0 + correlation) / 2.0
    distance = np.clip(distance, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance


def build_linkage(
    distance: np.ndarray,
    *,
    method: str = "average",
) -> np.ndarray:
    """Construct a SciPy linkage matrix from a distance matrix."""
    distance = np.asarray(distance, dtype=float)

    # Remove tiny numerical asymmetries introduced by floating-point
    # correlation and distance calculations.
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    condensed = squareform(
        distance,
        checks=True,
    )

    return linkage(
        condensed,
        method=method,
    )


def print_matrix(
    title: str,
    matrix: np.ndarray,
    labels: list[str],
) -> None:
    """Print a labelled square matrix."""
    print()
    print(title)
    print("-" * len(title))

    label_width = max(
        8,
        max(len(label) for label in labels) + 2,
    )

    print(
        " " * label_width
        + "".join(
            f"{label:>{label_width}}"
            for label in labels
        )
    )

    for label, row in zip(
        labels,
        matrix,
        strict=True,
    ):
        values = "".join(
            f"{value:>{label_width}.4f}"
            for value in row
        )
        print(f"{label:<{label_width}}{values}")


def print_linkage(
    title: str,
    hierarchy: np.ndarray,
    labels: list[str],
) -> None:
    """Print a hierarchical clustering linkage matrix."""
    print()
    print(title)
    print("-" * len(title))

    n_features = len(labels)

    cluster_names = {
        index: label
        for index, label in enumerate(labels)
    }

    for merge_index, row in enumerate(hierarchy):
        left = int(row[0])
        right = int(row[1])
        distance = float(row[2])
        count = int(row[3])

        left_name = cluster_names.get(
            left,
            f"cluster_{left}",
        )
        right_name = cluster_names.get(
            right,
            f"cluster_{right}",
        )

        new_index = n_features + merge_index
        new_name = f"({left_name}, {right_name})"
        cluster_names[new_index] = new_name

        print(
            f"merge {merge_index + 1}: "
            f"{left_name} + {right_name} "
            f"at distance {distance:.4f} "
            f"(size={count})"
        )

    print()
    print("Linkage matrix:")
    print(hierarchy)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze objective relationships and construct candidate "
            "MOO-informed Owen hierarchies."
        )
    )

    parser.add_argument(
        "--problem",
        choices=PROBLEM_CHOICES,
        default="river",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional background CSV path.",
    )
    parser.add_argument(
        "--linkage-method",
        choices=[
            "single",
            "complete",
            "average",
        ],
        default="average",
    )

    return parser.parse_args()


def main() -> None:
    """Run the objective-relationship analysis."""
    args = parse_args()

    input_path = args.input or default_background_path(
        args.problem,
        seed=args.seed,
        samples=args.samples,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Background dataset not found: {input_path}"
        )

    data = pl.read_csv(input_path)

    solution_columns = get_solution_columns(data)
    solution_values = (
        data.select(solution_columns)
        .to_numpy()
        .astype(float)
    )

    objective_labels = [
        column.removeprefix("s_")
        for column in solution_columns
    ]

    print("MOO objective relationship analysis")
    print("=" * 60)
    print(f"Dataset: {input_path}")
    print(f"Samples: {solution_values.shape[0]}")
    print(
        "Objectives: "
        + ", ".join(objective_labels)
    )
    print(
        f"Linkage method: {args.linkage_method}"
    )

    pearson = compute_pearson_correlation(
        solution_values
    )
    spearman = compute_spearman_correlation(
        solution_values
    )

    print_matrix(
        "Pearson correlation of solution objectives",
        pearson,
        objective_labels,
    )

    print_matrix(
        "Spearman correlation of solution objectives",
        spearman,
        objective_labels,
    )

    for correlation_name, correlation in (
        ("Pearson", pearson),
        ("Spearman", spearman),
    ):
        relationship = relationship_distance(
            correlation
        )
        conflict = conflict_distance(
            correlation
        )

        relationship_hierarchy = build_linkage(
            relationship,
            method=args.linkage_method,
        )
        conflict_hierarchy = build_linkage(
            conflict,
            method=args.linkage_method,
        )

        print_matrix(
            f"{correlation_name} relationship distance "
            "(1 - |rho|)",
            relationship,
            objective_labels,
        )

        print_linkage(
            f"{correlation_name} relationship hierarchy",
            relationship_hierarchy,
            objective_labels,
        )

        print_matrix(
            f"{correlation_name} conflict distance "
            "((1 + rho) / 2)",
            conflict,
            objective_labels,
        )

        print_linkage(
            f"{correlation_name} conflict hierarchy",
            conflict_hierarchy,
            objective_labels,
        )


if __name__ == "__main__":
    main()
