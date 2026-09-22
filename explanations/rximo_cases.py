"""Generalized R-XIMO case classification for multi-target explanations.

This module uses individual Owen attribution effects to classify a
reference point--solution pair according to a generalized version of the
nine R-XIMO explanation situations. The selected target objectives are
represented by their aggregate Owen attribution, while objectives outside
the target set are represented by their individual Owen values.

Direct coalition effects are used only for optional pairwise interaction
analysis. A meaningful interaction can refine an Owen-based individual
rival into a jointly considered rival pair.

The module contains no command-line interaction or optimization calls. Its
functions operate on already computed reference points, solutions, Owen
values, and interaction diagnostics.
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
class OwenEffect:
    """Store one individual Owen attribution effect."""

    member: str
    owen_value: float
    role: CoalitionRole
    is_target: bool


@dataclass(frozen=True)
class OwenEffectSummary:
    """Store Owen effects used for generalized R-XIMO reasoning."""

    effects: tuple[OwenEffect, ...]
    target_effect: float
    target_role: CoalitionRole
    external_supporters: tuple[OwenEffect, ...]
    external_rivals: tuple[OwenEffect, ...]
    strongest_external_supporter: OwenEffect | None
    weakest_external_supporter: OwenEffect | None
    strongest_external_rival: OwenEffect | None


@dataclass(frozen=True)
class RivalInteraction:
    """Store an interaction between two Owen-identified rivals."""

    members: tuple[str, str]
    interaction: float
    relative_interaction: float
    pair_contribution: float


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


def select_rival_interaction(
    interaction_summary: pl.DataFrame,
    external_rivals: tuple[OwenEffect, ...],
    selected_rival: str,
    *,
    interaction_tolerance: float = 1e-6,
    relative_threshold: float = 0.15,
) -> RivalInteraction | None:
    """Select the strongest meaningful interaction between rivals."""

    rival_members = {
        effect.member
        for effect in external_rivals
    }

    qualifying: list[RivalInteraction] = []

    for row in interaction_summary.to_dicts():
        members = tuple(row["members"])

        if not set(members).issubset(rival_members):
            continue

        if selected_rival not in members:
            continue

        interaction = float(row["interaction"])
        relative = float(row["relative_interaction"])

        if (
            interaction < -interaction_tolerance
            and relative >= relative_threshold
        ):
            qualifying.append(
                RivalInteraction(
                    members=members,
                    interaction=interaction,
                    relative_interaction=relative,
                    pair_contribution=float(
                        row["pair_contribution"]
                    ),
                )
            )

    if not qualifying:
        return None

    return min(
        qualifying,
        key=lambda candidate: candidate.interaction,
    )


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


def summarize_owen_effects(
    owen_summary: pl.DataFrame,
    target_symbols: list[str],
    tolerance: float = 1e-12,
) -> OwenEffectSummary:
    """Summarize individual effects derived from Owen values.

    Negative Owen values represent impairing effects on the
    higher-is-better target utility, while positive Owen values
    represent supporting effects.

    Args:
        owen_summary: Owen-value table containing ``input`` and
            ``owen_value`` columns.
        target_symbols: selected target objective symbols.
        tolerance: absolute threshold for neutral Owen effects.

    Returns:
        Individual Owen effects and the strongest or weakest
        effects needed for generalized R-XIMO reasoning.
    """
    target_set = target_input_members(target_symbols)

    effects: list[OwenEffect] = []

    for row in owen_summary.to_dicts():
        member = str(row["input"])
        value = float(row["owen_value"])

        if value > tolerance:
            role: CoalitionRole = "supporter"
        elif value < -tolerance:
            role = "rival"
        else:
            role = "neutral"

        effects.append(
            OwenEffect(
                member=member,
                owen_value=value,
                role=role,
                is_target=member in target_set,
            )
        )

    target_effects = [
        effect
        for effect in effects
        if effect.is_target
    ]

    target_effect = sum(
        effect.owen_value
        for effect in target_effects
    )

    if target_effect > tolerance:
        target_role: CoalitionRole = "supporter"
    elif target_effect < -tolerance:
        target_role = "rival"
    else:
        target_role = "neutral"

    supporters = [
        effect
        for effect in effects
        if effect.role == "supporter"
    ]

    rivals = [
        effect
        for effect in effects
        if effect.role == "rival"
    ]

    external_supporters = [
        effect
        for effect in supporters
        if not effect.is_target
    ]

    external_rivals = [
        effect
        for effect in rivals
        if not effect.is_target
    ]

    return OwenEffectSummary(
        effects=tuple(effects),
        target_effect=target_effect,
        target_role=target_role,
        external_supporters=tuple(
            sorted(
                external_supporters,
                key=lambda effect: effect.owen_value,
                reverse=True,
            )
        ),
        external_rivals=tuple(
            sorted(
                external_rivals,
                key=lambda effect: effect.owen_value,
            )
        ),
        strongest_external_supporter=(
            max(
                external_supporters,
                key=lambda effect: effect.owen_value,
            )
            if external_supporters
            else None
        ),
        weakest_external_supporter=(
            min(
                external_supporters,
                key=lambda effect: effect.owen_value,
            )
            if external_supporters
            else None
        ),
        strongest_external_rival=(
            min(
                external_rivals,
                key=lambda effect: effect.owen_value,
            )
            if external_rivals
            else None
        ),
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
    effects: OwenEffectSummary,
    tolerance: float = 1e-12,
) -> int:
    """Classify a generalized R-XIMO case from Owen effects.

    The selected target objectives are represented by their aggregate
    Owen attribution, while objectives outside the target set are
    represented by their individual Owen values.

    Args:
        reference_point_status: classified aspiration pattern.
        effects: Owen-effect summary for the selected target set.
        tolerance: numerical tolerance for comparing Owen effects.

    Returns:
        Generalized R-XIMO case number from 1 to 9.
    """
    target_effect = effects.target_effect

    strongest_external_supporter = (
        effects.strongest_external_supporter
    )
    weakest_external_supporter = (
        effects.weakest_external_supporter
    )
    strongest_external_rival = (
        effects.strongest_external_rival
    )

    target_is_strongest_supporter = (
        effects.target_role == "supporter"
        and (
            strongest_external_supporter is None
            or target_effect
            > strongest_external_supporter.owen_value + tolerance
        )
    )

    target_is_strongest_rival = (
        effects.target_role == "rival"
        and (
            strongest_external_rival is None
            or target_effect
            < strongest_external_rival.owen_value - tolerance
        )
    )

    target_is_weakest_supporter = (
        effects.target_role == "supporter"
        and (
            weakest_external_supporter is None
            or target_effect
            < weakest_external_supporter.owen_value - tolerance
        )
    )

    if reference_point_status == "too_demanding":
        return 2 if target_is_strongest_rival else 1

    if reference_point_status == "pessimistic":
        return 3 if target_is_weakest_supporter else 4

    has_supporting_effect = (
        effects.target_role == "supporter"
        or bool(effects.external_supporters)
    )

    has_impairing_effect = (
        effects.target_role == "rival"
        or bool(effects.external_rivals)
    )

    if not has_supporting_effect:
        return 5

    if not has_impairing_effect:
        return 6

    if target_is_strongest_rival:
        return 8

    if target_is_strongest_supporter:
        return 9

    return 7


def select_rival_for_case(
    case_number: int,
    effects: OwenEffectSummary,
) -> OwenEffect | None:
    """Select the external Owen effect to relax for an R-XIMO case.

    Cases seeking an impairing effect select the strongest external
    Owen rival. Cases corresponding to an absence of impairing effects
    select the weakest external supporter, preserving the semantic
    intent of the original R-XIMO recommendation logic.

    Args:
        case_number: generalized R-XIMO case number.
        effects: summarized Owen effects.

    Returns:
        Selected external Owen effect, or ``None`` when no suitable
        external effect exists.
    """
    if case_number in {1, 2, 5, 7, 8, 9}:
        return effects.strongest_external_rival

    if case_number in {3, 4, 6}:
        return effects.weakest_external_supporter

    return None


def rank_rivals_for_case(
    case_number: int,
    effects: OwenEffectSummary,
) -> list[OwenEffect]:
    """Rank external Owen effects for preference adjustment.

    Cases 1, 2, 5, 7, 8, and 9 rank impairing external effects from
    strongest to weakest. Cases 3, 4, and 6 rank supporting external
    effects from weakest to strongest.

    Args:
        case_number: generalized R-XIMO case number.
        effects: summarized Owen effects.

    Returns:
        External Owen effects ordered by preference for adjustment.
    """
    if case_number in {1, 2, 5, 7, 8, 9}:
        return list(effects.external_rivals)

    if case_number in {3, 4, 6}:
        return sorted(
            effects.external_supporters,
            key=lambda effect: effect.owen_value,
        )

    return []


def select_actionable_rival_for_case(
    case_number: int,
    effects: OwenEffectSummary,
    reference_point_min: np.ndarray,
    nadir_min: np.ndarray,
    objective_symbols: list[str],
) -> tuple[
    OwenEffect | None,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Select the strongest eligible Owen-based rival that can be impaired.

    Candidates are considered according to the generalized R-XIMO
    ranking. Candidates whose aspiration is already at or beyond the
    nadir are skipped.

    Args:
        case_number: generalized R-XIMO case number.
        effects: summarized Owen effects.
        reference_point_min: reference point in minimization orientation.
        nadir_min: nadir vector in minimization orientation.
        objective_symbols: objective symbols in problem order.

    Returns:
        Selected Owen effect, actionable member tuple, and
        non-actionable member tuple.
    """
    ranked_candidates = rank_rivals_for_case(
        case_number,
        effects,
    )

    for candidate in ranked_candidates:
        members = (candidate.member,)

        actionable = actionable_rival_members(
            members,
            reference_point_min,
            nadir_min,
            objective_symbols,
        )

        if not actionable:
            continue

        non_actionable = tuple(
            member
            for member in members
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
    rival: OwenEffect | None,
    effects: OwenEffectSummary,
    interaction: RivalInteraction | None = None,
) -> str:
    """Generate a verbal explanation for a generalized R-XIMO case.

    The explanation is based primarily on Owen attribution effects.
    When a meaningful interaction is identified between the selected
    individual rival and another Owen-identified rival, the interaction
    is reported as an additional joint effect.

    Args:
        case_number: generalized R-XIMO case number.
        target_members: selected target reference point components.
        rival: individual external Owen effect selected by the case logic.
        effects: summarized Owen effects.
        interaction: optional meaningful interaction involving the selected rival.

    Returns:
        Decision-maker-facing explanation of the selected case.
    """
    target_text = format_members(target_members)

    rival_text = (
        format_members((rival.member,))
        if rival is not None
        else "no external alternative"
    )

    strongest_external_rival_text = (
        format_members(
            (effects.strongest_external_rival.member,)
        )
        if effects.strongest_external_rival is not None
        else "none"
    )

    explanations = {
        1: (
            "The solution is worse than the aspiration level in every "
            "objective, so the reference point appears too demanding. "
            f"However, no external aspiration level was identified as "
            f"having an impairing effect on {target_text}."
        ) if strongest_external_rival_text == "none" else (
            "The solution is worse than the aspiration level in every "
            "objective, so the reference point appears too demanding. "
            f"The strongest impairing effect for "
            f"{target_text} is {strongest_external_rival_text}."
        ),
        2: (
            "The solution is worse than the aspiration level in every "
            f"objective, and the aspiration levels of {target_text} "
            "collectively have the strongest impairing effect. "
            f"The strongest external alternative is {rival_text}."
        ),
        3: (
            "The solution is better than the aspiration level in every "
            "objective, so the reference point appears pessimistic. "
            f"The aspiration levels of {target_text} collectively have "
            "the weakest supporting effect. Among the objectives "
            f"outside the selected target set, {rival_text} has the "
            "weakest supporting effect."
        ),
        4: (
            "The solution is better than the aspiration level in every "
            "objective, so the reference point appears pessimistic. "
            "Among the objectives outside the selected target set, "
            f"{rival_text} has the weakest supporting effect."
        ),
        5: (
            f"No component supports {target_text}. The strongest "
            f"external impairing effect is {strongest_external_rival_text}."
        ),
        6: (
            f"No component impairs {target_text}. Among the objectives "
            f"outside the selected target set, {rival_text} has the "
            "weakest supporting effect."
        ),
        7: (
            f"The effects for {target_text} contain both supporting "
            "and impairing influences. The strongest external impairing "
            f"effect is {strongest_external_rival_text}."
        ),
        8: (
            f"The aspiration levels of {target_text} collectively have "
            "the strongest impairing effect. The strongest external "
            f"rival is {rival_text}."
        ),
        9: (
            f"The aspiration levels of {target_text} collectively have "
            "the strongest supporting effect. The strongest external "
            f"impairing effect is {strongest_external_rival_text}."
        ),
    }

    explanation = explanations[case_number]

    if interaction is not None:
        interaction_text = format_members(
            interaction.members
        )
        explanation += (
            f" In addition, {interaction_text} exhibits an additional "
            "joint impairing effect beyond the corresponding individual "
            "coalition effects."
        )

    return explanation


def generate_suggestion(
    actionable_targets: tuple[str, ...],
    actionable_rivals: tuple[str, ...],
) -> str:
    """Generate an actionable preference-change suggestion."""
    if actionable_targets:
        target_text = format_members(actionable_targets)

        if actionable_rivals:
            rival_text = format_members(actionable_rivals)

            target_word = (
                "level"
                if len(actionable_targets) == 1
                else "levels"
            )
            rival_word = (
                "level"
                if len(actionable_rivals) == 1
                else "levels"
            )

            return (
                f"Try improving the aspiration {target_word} for "
                f"{target_text} and relaxing the aspiration "
                f"{rival_word} for {rival_text}."
            )

        target_word = (
            "level"
            if len(actionable_targets) == 1
            else "levels"
        )

        return (
            f"Try improving the aspiration {target_word} for "
            f"{target_text}."
        )

    if actionable_rivals:
        rival_text = format_members(actionable_rivals)

        rival_word = (
            "level"
            if len(actionable_rivals) == 1
            else "levels"
        )

        return (
            f"Try relaxing the aspiration {rival_word} for "
            f"{rival_text}."
        )

    return "No further aspiration-level adjustment was identified."


def build_rximo_suggestion(
    reference_point_status: ReferencePointStatus,
    owen_summary: pl.DataFrame,
    interaction_summary: pl.DataFrame,
    target_symbols: list[str],
    reference_point_min: np.ndarray,
    ideal_min: np.ndarray,
    nadir_min: np.ndarray,
    objective_symbols: list[str],
    interaction_tolerance: float = 1e-6,
    interaction_relative_threshold: float = 0.15,
) -> RXIMOSuggestion:
    """Build a generalized R-XIMO suggestion from Owen effects.

    Owen values provide the primary attribution effects used for case
    classification and individual rival selection. Pairwise coalition
    interactions are used as a secondary refinement when multiple
    external objectives have impairing Owen effects.

    Args:
        reference_point_status: classified aspiration pattern.
        owen_summary: individual Owen-value summary.
        interaction_summary: pairwise coalition-interaction diagnostics.
        target_symbols: selected target objective symbols.
        reference_point_min: reference point in common minimization orientation.
        ideal_min: ideal objective vector in common minimization orientation.
        nadir_min: nadir objective vector in common minimization orientation.
        objective_symbols: objective symbols in problem order.
        interaction_tolerance: minimum absolute negative interaction required
            for promoting an individual rival to an interacting pair.
        interaction_relative_threshold: minimum relative interaction strength
            required for promoting an individual rival to an interacting pair.

    Returns:
        Selected generalized R-XIMO case, explanation, and actionable
        preference-change suggestion.
    """
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

    effects = summarize_owen_effects(
        owen_summary,
        target_symbols,
    )

    number = classify_rximo_case(
        reference_point_status,
        effects,
    )

    explanatory_rival = select_rival_for_case(
        number,
        effects,
    )

    (
        action_rival,
        actionable_rivals,
        non_actionable_rivals,
    ) = select_actionable_rival_for_case(
        number,
        effects,
        reference_point_min,
        nadir_min,
        objective_symbols,
    )

    """print("\nOWEN-BASED R-XIMO DEBUG")
    print("-" * 50)
    print(f"Target effect: {effects.target_effect:+.6f}")
    print(f"Target role: {effects.target_role}")

    print("External supporters:")
    for effect in effects.external_supporters:
        print(
            f"  {effect.member}: "
            f"{effect.owen_value:+.6f}"
        )

    print("External rivals:")
    for effect in effects.external_rivals:
        print(
            f"  {effect.member}: "
            f"{effect.owen_value:+.6f}"
        )

    print(f"R-XIMO case: {number}")

    print(
        "Explanatory rival:",
        explanatory_rival.member
        if explanatory_rival is not None
        else "none",
    )

    print(
        "Action rival:",
        action_rival.member
        if action_rival is not None
        else "none",
    )"""

    interaction = None
    selected_action_rival_members: tuple[str, ...] = ()

    if action_rival is not None:
        selected_action_rival_members = (
            action_rival.member,
        )

        # Interaction refinement applies only when the selected
        # adjustment is an actual impairing Owen effect. Cases that
        # deliberately relax a weak supporter are not promoted to
        # interacting rival coalitions.
        if action_rival.role == "rival":
            """print("Eligible rival interactions:")

            external_rival_members = {
                effect.member
                for effect in effects.external_rivals
            }

            for row in interaction_summary.to_dicts():
                members = tuple(row["members"])

                if (
                    set(members).issubset(external_rival_members)
                    and action_rival.member in members
                ):
                    print(
                        f"  {members}: "
                        f"interaction={float(row['interaction']):+.6f}, "
                        f"relative={float(row['relative_interaction']):.3f}"
                    )"""

            interaction = select_rival_interaction(
                interaction_summary,
                effects.external_rivals,
                selected_rival=action_rival.member,
                interaction_tolerance=interaction_tolerance,
                relative_threshold=interaction_relative_threshold,
            )

            if interaction is not None:
                selected_action_rival_members = (
                    interaction.members
                )

                actionable_rivals = actionable_rival_members(
                    selected_action_rival_members,
                    reference_point_min,
                    nadir_min,
                    objective_symbols,
                )

                non_actionable_rivals = tuple(
                    member
                    for member in selected_action_rival_members
                    if member not in actionable_rivals
                )

    """if interaction is not None:
        print(
            "Promoted rival interaction:",
            interaction.members,
        )
        print(
            f"Interaction: "
            f"{interaction.interaction:+.6f}"
        )
        print(
            f"Relative interaction: "
            f"{interaction.relative_interaction:.3f}"
        )
    else:
        print("Promoted rival interaction: none")"""

    strongest_supporter_members: tuple[str, ...] = ()

    if effects.strongest_external_supporter is not None:
        strongest_supporter_members = (
            effects.strongest_external_supporter.member,
        )

    strongest_rival_members: tuple[str, ...] = ()

    if effects.strongest_external_rival is not None:
        strongest_rival_members = (
            effects.strongest_external_rival.member,
        )

    return RXIMOSuggestion(
        case_number=number,
        case_name=case_name(number),
        reference_point_status=reference_point_status,
        target_members=target_members,
        actionable_target_members=actionable_targets,
        non_actionable_target_members=non_actionable_targets,
        rival_members=(
            (explanatory_rival.member,)
            if explanatory_rival is not None
            else ()
        ),
        selected_action_rival_members=selected_action_rival_members,
        actionable_rival_members=actionable_rivals,
        non_actionable_rival_members=non_actionable_rivals,
        strongest_supporter_members=strongest_supporter_members,
        strongest_rival_members=strongest_rival_members,
        explanation=generate_explanation(
            number,
            target_members,
            explanatory_rival,
            effects,
            interaction,
        ),
        suggestion=generate_suggestion(
            actionable_targets,
            actionable_rivals,
        ),
    )
