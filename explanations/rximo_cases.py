"""Generalized R-XIMO case classification for multi-target explanations.

This module converts direct input-coalition contributions into supporting,
impairing, and neutral effects and uses them to classify a reference
point--solution pair according to a generalized version of the nine R-XIMO
explanation situations.

The module contains no command-line interaction or optimization calls. Its
functions operate on already computed reference points, solutions, and
coalition-contribution tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl


ReferencePointStatus = Literal["too_demanding", "pessimistic", "mixed"]

CoalitionCategory = Literal[
    "full_target",
    "target_subset",
    "external",
    "mixed",
]

CoalitionRole = Literal[
    "supporter",
    "rival",
    "neutral",
]


@dataclass(frozen=True)
class CoalitionCandidate:
    """Store one categorized input-coalition contribution.

    Attributes:
        members: reference point component symbols in the coalition.
        member_set: coalition members represented as an immutable set.
        size: number of members in the coalition.
        category: relationship between the coalition and the target coalition.
        contribution: target-utility contribution relative to the empty
            coalition baseline.
        role: qualitative interpretation of the contribution sign.
    """
    members: tuple[str, ...]
    member_set: frozenset[str]
    size: int
    category: CoalitionCategory
    contribution: float
    role: CoalitionRole


@dataclass(frozen=True)
class EffectSummary:
    """Store the retained supporting and impairing coalition effects."""
    strongest_supporter: CoalitionCandidate | None
    strongest_rival: CoalitionCandidate | None
    weakest_supporter: CoalitionCandidate | None
    weakest_external_supporter: CoalitionCandidate | None
    alternative_rival: CoalitionCandidate | None
    retained_candidates: tuple[CoalitionCandidate, ...]


@dataclass(frozen=True)
class RXIMOSuggestion:
    """Store a generalized R-XIMO case, explanation, and preference suggestion."""
    case_number: int
    case_name: str
    reference_point_status: ReferencePointStatus
    target_members: tuple[str, ...]
    actionable_target_members: tuple[str, ...]
    non_actionable_target_members: tuple[str, ...]
    rival_members: tuple[str, ...]
    selected_action_rival_members: tuple[str, ...]
    actionable_rival_members: tuple[str, ...]
    non_actionable_rival_members: tuple[str, ...]
    strongest_supporter_members: tuple[str, ...]
    strongest_rival_members: tuple[str, ...]
    explanation: str
    suggestion: str


def clean_symbol(symbol: str) -> str:
    """Remove a reference point or solution prefix from a symbol.

    Args:
        symbol (str): an objective, reference point, or solution symbol.

    Returns:
        str: the underlying objective symbol.
    """
    return symbol.removeprefix("r_").removeprefix("s_")


def format_members(members: tuple[str, ...] | list[str]) -> str:
    """Format coalition members for decision-maker-facing output.

    Args:
        members (tuple[str, ...] | list[str]): coalition member symbols.

    Returns:
        str: a cleaned singleton, braced coalition, or ``"none"``.
    """
    cleaned = [clean_symbol(member) for member in members]
    if not cleaned:
        return "none"
    if len(cleaned) == 1:
        return cleaned[0]
    return "{" + ", ".join(cleaned) + "}"


def target_input_members(target_symbols: list[str]) -> frozenset[str]:
    """Map target output symbols to their reference point input symbols.

    Args:
        target_symbols (list[str]): target solution symbols.

    Returns:
        frozenset[str]: corresponding reference point symbols.
    """
    return frozenset(
        f"r_{clean_symbol(target)}"
        for target in target_symbols
    )


def objective_index(
    symbol: str,
    objective_symbols: list[str],
) -> int:
    """Return the objective index corresponding to an R-XIMO symbol.

    Args:
        symbol (str): objective, reference point, or solution symbol.
        objective_symbols (list[str]): objective symbols in problem order.

    Returns:
        int: index of the corresponding objective.

    Raises:
        ValueError: if the symbol does not correspond to an objective.
    """
    cleaned = clean_symbol(symbol)
    cleaned_objectives = [
        clean_symbol(objective)
        for objective in objective_symbols
    ]
    return cleaned_objectives.index(cleaned)


def actionable_target_members(
    target_members: tuple[str, ...],
    reference_point_min: np.ndarray,
    ideal_min: np.ndarray,
    objective_symbols: list[str],
) -> tuple[str, ...]:
    """Return target aspirations that can still be improved.

    A target is actionable only when its aspiration is worse than its ideal
    value in common minimization orientation. Aspirations already at or beyond
    the ideal are omitted from the preference-change suggestion.

    Args:
        target_members (tuple[str, ...]): target reference point symbols.
        reference_point_min (np.ndarray): reference point in common minimization
            orientation.
        ideal_min (np.ndarray): ideal objective vector in common minimization
            orientation.
        objective_symbols (list[str]): objective symbols in problem order.

    Returns:
        tuple[str, ...]: target members for which further aspiration improvement
            is meaningful.
    """
    actionable = []

    for target in target_members:
        index = objective_index(
            target,
            objective_symbols,
        )

        aspiration = float(reference_point_min[index])
        ideal = float(ideal_min[index])

        if aspiration > ideal:
            actionable.append(target)

    return tuple(actionable)


def actionable_rival_members(
    rival_members: tuple[str, ...],
    reference_point_min: np.ndarray,
    nadir_min: np.ndarray,
    objective_symbols: list[str],
) -> tuple[str, ...]:
    """Return rival aspirations that can still be impaired.

    A rival is actionable only when its aspiration is better than its nadir
    value in common minimization orientation. Aspirations already at or beyond
    the nadir are omitted from the preference-change suggestion.

    Args:
        rival_members: rival reference point component symbols.
        reference_point_min: reference point in common minimization orientation.
        nadir_min: nadir objective vector in common minimization orientation.
        objective_symbols: objective symbols in problem order.

    Returns:
        Rival members for which further aspiration impairment is meaningful.
    """
    actionable = []

    for rival in rival_members:
        index = objective_index(
            rival,
            objective_symbols,
        )

        aspiration = float(reference_point_min[index])
        nadir = float(nadir_min[index])

        if aspiration < nadir:
            actionable.append(rival)

    return tuple(actionable)


def classify_reference_point_status(
    reference_point_min: np.ndarray,
    solution_min: np.ndarray,
    tolerance: float = 1e-8,
) -> ReferencePointStatus:
    """Classify the aspiration pattern in common minimization orientation."""
    difference = solution_min - reference_point_min

    if np.all(difference > tolerance):
        return "too_demanding"
    if np.all(difference < -tolerance):
        return "pessimistic"
    return "mixed"


def coalition_rows_to_candidates(
    coalition_summary: pl.DataFrame,
    target_symbols: list[str],
    tolerance: float = 1e-12,
) -> list[CoalitionCandidate]:
    """Convert coalition-table rows into generalized R-XIMO candidates.

    Coalitions are categorized relative to the complete target coalition as the
    full target, a target subset, an external coalition, or a mixed coalition.
    Their direct contribution determines whether they are supporting, impairing,
    or neutral.

    Args:
        coalition_summary: direct coalition contribution table.
        target_symbols: selected target output symbols.
        tolerance: threshold for neutral contributions. Defaults to 1e-12.

    Returns:
        Categorized coalition candidates.
    """
    target_set = target_input_members(target_symbols)
    candidates: list[CoalitionCandidate] = []

    for row in coalition_summary.to_dicts():
        members = tuple(row["coalition"])
        member_set = frozenset(members)
        contribution = float(row["contribution"])

        if member_set == target_set:
            category: CoalitionCategory = "full_target"
        elif member_set < target_set:
            category = "target_subset"
        elif member_set.isdisjoint(target_set):
            category = "external"
        else:
            category = "mixed"

        if contribution > tolerance:
            role: CoalitionRole = "supporter"
        elif contribution < -tolerance:
            role = "rival"
        else:
            role = "neutral"

        candidates.append(
            CoalitionCandidate(
                members=members,
                member_set=member_set,
                size=len(members),
                category=category,
                contribution=contribution,
                role=role,
            )
        )

    return candidates


def best_proper_subset(
    candidate: CoalitionCandidate,
    candidates: list[CoalitionCandidate],
) -> CoalitionCandidate | None:
    """Find the strongest proper subset with the same category and role.

    Args:
        candidate: candidate whose subsets are considered.
        candidates: available coalition candidates.

    Returns:
        Best matching proper subset, or ``None`` when none exists.
    """
    subsets = [
        other
        for other in candidates
        if other.category == candidate.category
        and other.role == candidate.role
        and other.member_set < candidate.member_set
    ]

    if not subsets:
        return None

    if candidate.role == "supporter":
        return max(
            subsets,
            key=lambda row: row.contribution,
        )

    return min(
        subsets,
        key=lambda row: row.contribution,
    )


def candidate_advantage(
    candidate: CoalitionCandidate,
    candidates: list[CoalitionCandidate],
) -> tuple[float, float]:
    """Measure the incremental effect over the candidate's best subset.

    Args:
        candidate: coalition candidate to assess.
        candidates: candidates used to find a proper subset.

    Returns:
        Absolute and relative contribution advantages.
    """
    subset = best_proper_subset(
        candidate,
        candidates,
    )

    if subset is None:
        return (
            abs(candidate.contribution),
            float("inf"),
        )

    advantage = (
        abs(candidate.contribution)
        - abs(subset.contribution)
    )
    subset_effect = abs(subset.contribution)

    if np.isclose(subset_effect, 0.0):
        relative = (
            float("inf")
            if advantage > 0
            else 0.0
        )
    else:
        relative = advantage / subset_effect

    return advantage, relative


def remove_redundant_external_candidates(
    candidates: list[CoalitionCandidate],
    minimum_relative_advantage: float,
) -> list[CoalitionCandidate]:
    """Prefer smaller external coalitions unless a larger one adds enough.

    Args:
        candidates: categorized coalition candidates.
        minimum_relative_advantage: minimum relative improvement required for
            retaining a larger external coalition.

    Returns:
        Retained external coalition candidates.
    """
    external = [
        candidate
        for candidate in candidates
        if candidate.category == "external"
    ]

    retained: list[CoalitionCandidate] = []

    for candidate in external:
        if candidate.size == 1:
            retained.append(candidate)
            continue

        advantage, relative = candidate_advantage(
            candidate,
            external,
        )

        if (
            advantage > 0
            and relative >= minimum_relative_advantage
        ):
            retained.append(candidate)

    return retained


def summarize_effects(
    candidates: list[CoalitionCandidate],
    target_set: frozenset[str],
    coalition_advantage_threshold: float,
) -> EffectSummary:
    """Summarize retained effects for generalized case classification.

    Only the complete target coalition and non-redundant external coalitions are
    retained. Target subsets and mixed coalitions are intentionally excluded
    from the final supporter--rival comparison.

    Args:
        candidates: categorized coalition candidates.
        target_set: complete target coalition in input symbols.
        coalition_advantage_threshold: minimum relative advantage for retaining
            larger external coalitions.

    Returns:
        Strongest and weakest retained effects used by case classification and
        rival selection.
    """
    external = remove_redundant_external_candidates(
        candidates,
        coalition_advantage_threshold,
    )

    full_target = next(
        (
            candidate
            for candidate in candidates
            if candidate.member_set == target_set
        ),
        None,
    )

    retained = list(external)

    if full_target is not None:
        retained.append(full_target)

    supporters = [
        candidate
        for candidate in retained
        if candidate.role == "supporter"
    ]

    rivals = [
        candidate
        for candidate in retained
        if candidate.role == "rival"
    ]

    strongest_supporter = (
        max(
            supporters,
            key=lambda row: row.contribution,
        )
        if supporters
        else None
    )

    strongest_rival = (
        min(
            rivals,
            key=lambda row: row.contribution,
        )
        if rivals
        else None
    )

    weakest_supporter = (
        min(
            supporters,
            key=lambda row: row.contribution,
        )
        if supporters
        else None
    )

    external_rivals = [
        candidate
        for candidate in rivals
        if candidate.category == "external"
    ]

    alternative_rival = (
        min(
            external_rivals,
            key=lambda row: row.contribution,
        )
        if external_rivals
        else None
    )

    external_supporters = [
        candidate
        for candidate in supporters
        if candidate.category == "external"
    ]

    weakest_external_supporter = (
        min(
            external_supporters,
            key=lambda row: row.contribution,
        )
        if external_supporters
        else None
    )

    return EffectSummary(
        strongest_supporter=strongest_supporter,
        strongest_rival=strongest_rival,
        weakest_supporter=weakest_supporter,
        weakest_external_supporter=weakest_external_supporter,
        alternative_rival=alternative_rival,
        retained_candidates=tuple(retained),
    )


def classify_rximo_case(
    reference_point_status: ReferencePointStatus,
    effects: EffectSummary,
    target_set: frozenset[str],
) -> int:
    """Generalized implementation of the nine Table 1 situations."""
    best = effects.strongest_supporter
    worst = effects.strongest_rival
    weakest = effects.weakest_supporter

    target_is_best = (
        best is not None
        and best.member_set == target_set
    )

    target_is_worst = (
        worst is not None
        and worst.member_set == target_set
    )

    target_is_weakest_supporter = (
        weakest is not None
        and weakest.member_set == target_set
    )

    if reference_point_status == "too_demanding":
        return 2 if target_is_worst else 1

    if reference_point_status == "pessimistic":
        return 3 if target_is_weakest_supporter else 4

    if best is None:
        return 5

    if worst is None:
        return 6

    if target_is_worst:
        return 8

    if target_is_best:
        return 9

    return 7


def select_rival_for_case(
    case_number: int,
    effects: EffectSummary,
) -> CoalitionCandidate | None:
    """Select the explanatory external coalition for a classified R-XIMO case.

    This selection is based only on the retained coalition effects and the
    generalized R-XIMO case. Aspiration bounds are intentionally not considered
    here because this coalition is used to explain the case, not to determine
    whether a preference adjustment is still actionable.

    Args:
        case_number: generalized R-XIMO case number.
        effects: retained coalition-effect summary.

    Returns:
        External coalition used in the R-XIMO explanation, or ``None`` when no
        suitable external alternative exists.
    """
    if case_number in {1, 5, 7, 9}:
        rival = effects.strongest_rival

        if (
            rival is not None
            and rival.category == "external"
        ):
            return rival

        return effects.alternative_rival

    if case_number in {2, 8}:
        return effects.alternative_rival

    if case_number in {3, 4, 6}:
        return effects.weakest_external_supporter

    return None


def rank_rivals_for_case(
    case_number: int,
    effects: EffectSummary,
) -> list[CoalitionCandidate]:
    """Rank external preference-change candidates for an R-XIMO case.

    Cases 1, 2, 5, 7, 8, and 9 seek the strongest external impairing
    coalition. Cases 3, 4, and 6 seek the weakest external supporting
    coalition.

    Args:
        case_number: generalized R-XIMO case number.
        effects: retained coalition-effect summary.

    Returns:
        External candidates ordered from most preferred to least preferred
        for the corresponding case.
    """
    external = [
        candidate
        for candidate in effects.retained_candidates
        if candidate.category == "external"
    ]

    if case_number in {1, 2, 5, 7, 8, 9}:
        return sorted(
            (
                candidate
                for candidate in external
                if candidate.role == "rival"
            ),
            key=lambda row: row.contribution,
        )

    if case_number in {3, 4, 6}:
        return sorted(
            (
                candidate
                for candidate in external
                if candidate.role == "supporter"
            ),
            key=lambda row: row.contribution,
        )

    return []


def select_actionable_rival_for_case(
    case_number: int,
    effects: EffectSummary,
    reference_point_min: np.ndarray,
    nadir_min: np.ndarray,
    objective_symbols: list[str],
) -> tuple[
    CoalitionCandidate | None,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Select the strongest rival candidate with an available adjustment.

    Returns:
        Selected coalition, its actionable members, and members already at or
        beyond nadir.
    """
    ranked_candidates = rank_rivals_for_case(
        case_number,
        effects,
    )

    for candidate in ranked_candidates:
        actionable = actionable_rival_members(
            candidate.members,
            reference_point_min,
            nadir_min,
            objective_symbols,
        )

        if not actionable:
            continue

        non_actionable = tuple(
            member
            for member in candidate.members
            if member not in actionable
        )

        return (
            candidate,
            actionable,
            non_actionable,
        )

    return None, (), ()


def case_name(case_number: int) -> str:
    """Return a descriptive name for a generalized R-XIMO case.

    Args:
        case_number (int): case number from 1 to 9.

    Returns:
        str: descriptive case name.
    """
    names = {
        1: "too-demanding reference point",
        2: "too-demanding target coalition",
        3: "pessimistic target coalition",
        4: "pessimistic reference point",
        5: "no supporting effect",
        6: "no impairing effect",
        7: "ordinary supporter-rival pattern",
        8: "target coalition is strongest rival",
        9: "target coalition is strongest supporter",
    }
    return names[case_number]


def generate_explanation(
    case_number: int,
    target_members: tuple[str, ...],
    rival: CoalitionCandidate | None,
    effects: EffectSummary,
) -> str:
    """Generate a verbal explanation for a classified R-XIMO case.

    Args:
        case_number (int): generalized R-XIMO case number.
        target_members (tuple[str, ...]): complete target coalition.
        rival (dict | None): selected external alternative.
        effects (EffectSummary): retained effect summary.

    Returns:
        str: decision-maker-facing explanation of the selected case.
    """
    target_text = format_members(target_members)

    rival_text = (
        format_members(rival.members)
        if rival is not None
        else "no external alternative"
    )

    best_text = (
        format_members(
            effects.strongest_supporter.members
        )
        if effects.strongest_supporter is not None
        else "none"
    )

    worst_text = (
        format_members(
            effects.strongest_rival.members
        )
        if effects.strongest_rival is not None
        else "none"
    )

    explanations = {
        1: (
            "The solution is worse than the aspiration level in every "
            "objective, so the reference point appears too demanding. "
            f"The strongest impairing effect for {target_text} is {worst_text}."
        ),
        2: (
            "The solution is worse than the aspiration level in every "
            f"objective, and {target_text} itself has the strongest impairing "
            f"effect. The external alternative is {rival_text}."
        ),
        3: (
            "The solution is better than the aspiration level in every "
            "objective, so the reference point appears pessimistic. "
            f"The complete target coalition {target_text} has the weakest "
            "supporting coalition-level effect. Among the objectives outside "
            f"the selected target coalition, {rival_text} has the weakest "
            "supporting effect."
        ),
        4: (
            "The solution is better than the aspiration level in every "
            "objective, so the reference point appears pessimistic. "
            "Among the objectives outside the selected target coalition, "
            f"{rival_text} has the weakest supporting coalition-level effect."
        ),
        5: (
            f"No retained component supports {target_text}. The strongest "
            f"impairing effect is {worst_text}."
        ),
        6: (
            f"No retained component impairs {target_text}. The weakest "
            f"supporting effect is {rival_text}."
        ),
        7: (
            f"The selected target coalition {target_text} is most supported "
            f"by {best_text} and most impaired by {worst_text}."
        ),
        8: (
            f"The target coalition {target_text} is most impaired by its own "
            f"aspiration levels. The strongest external rival is {rival_text}."
        ),
        9: (
            f"The target coalition {target_text} is most supported by its own "
            f"aspiration levels and most impaired by {worst_text}."
        ),
    }
    return explanations[case_number]


def generate_suggestion(
    actionable_targets: tuple[str, ...],
    actionable_rivals: tuple[str, ...],
) -> str:
    """Generate an actionable preference-change suggestion."""
    if actionable_targets:
        target_text = format_members(actionable_targets)

        if actionable_rivals:
            rival_text = format_members(actionable_rivals)
            return (
                f"Try improving the aspiration levels for {target_text} and "
                f"impairing the aspiration levels for {rival_text}."
            )

        return (
            f"Try improving the aspiration levels for {target_text}."
        )

    if actionable_rivals:
        rival_text = format_members(actionable_rivals)
        return (
            f"Try impairing the aspiration levels for {rival_text}."
        )

    return "No further aspiration-level adjustment was identified."


def build_rximo_suggestion(
    reference_point_status: ReferencePointStatus,
    candidates: list[CoalitionCandidate],
    target_symbols: list[str],
    coalition_advantage_threshold: float,
    reference_point_min: np.ndarray,
    ideal_min: np.ndarray,
    nadir_min: np.ndarray,
    objective_symbols: list[str],
) -> RXIMOSuggestion:
    """Build a complete generalized R-XIMO suggestion.

    Args:
        reference_point_status (ReferencePointStatus): classified aspiration
        pattern.
        candidates (list[CoalitionCandidate]): categorized coalition candidates.
        target_symbols (list[str]): selected target output symbols.
        coalition_advantage_threshold (float): redundancy threshold for larger
        external coalitions.
        reference_point_min (np.ndarray): reference point in common minimization
            orientation.
        ideal_min (np.ndarray): ideal objective vector in common minimization
            orientation.
        nadir_min (np.ndarray): nadir objective vector in common minimization
            orientation.
        objective_symbols (list[str]): objective symbols in problem order.

    Returns:
        RXIMOSuggestion: selected case, relevant coalitions, explanation, and
        preference-change suggestion.
    """
    target_set = target_input_members(target_symbols)
    target_members = tuple(
        f"r_{clean_symbol(target)}"
        for target in target_symbols
    )
    actionable_targets = actionable_target_members(
        target_members,
        reference_point_min,
        ideal_min,
        objective_symbols,
    )

    non_actionable_targets = tuple(
        target
        for target in target_members
        if target not in actionable_targets
    )

    effects = summarize_effects(
        candidates,
        target_set,
        coalition_advantage_threshold,
    )
    number = classify_rximo_case(
        reference_point_status,
        effects,
        target_set,
    )

    explanatory_rival = select_rival_for_case(
        number,
        effects,
    )

    action_rival, actionable_rivals, non_actionable_rivals = select_actionable_rival_for_case(
        number,
        effects,
        reference_point_min,
        nadir_min,
        objective_symbols,
    )

    return RXIMOSuggestion(
        case_number=number,
        case_name=case_name(number),
        reference_point_status=reference_point_status,
        target_members=target_members,
        actionable_target_members=actionable_targets,
        non_actionable_target_members=non_actionable_targets,
        rival_members=(
            tuple(explanatory_rival.members)
            if explanatory_rival is not None
            else ()
        ),
        selected_action_rival_members=(
            tuple(action_rival.members)
            if action_rival is not None
            else ()
        ),
        actionable_rival_members=actionable_rivals,
        non_actionable_rival_members=non_actionable_rivals,
        strongest_supporter_members=(
            tuple(effects.strongest_supporter.members)
            if effects.strongest_supporter is not None
            else ()
        ),
        strongest_rival_members=(
            tuple(effects.strongest_rival.members)
            if effects.strongest_rival is not None
            else ()
        ),
        explanation=generate_explanation(
            number,
            target_members,
            explanatory_rival,
            effects,
        ),
        suggestion=generate_suggestion(
            actionable_targets,
            actionable_rivals,
        ),
    )
