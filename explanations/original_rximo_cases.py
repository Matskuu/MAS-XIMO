"""Isolated single-target R-XIMO (Misitano et al., 2022) case classifier.

Inputs MUST be in minimization orientation. SHAP entries explain the
minimized target objective: positive = impairing, negative = improving.
This module does not import or modify generalized multi-target Owen rules.

Source: https://doi.org/10.1007/s10458-022-09577-3, Sec. 3.3,
Algorithm 1 and Table 1. Text is paraphrased, not copied.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OriginalRXIMOResult:
    case_number: int
    case_name: str
    target_index: int
    rival_index: int
    best_effect: int | None
    worst_effect: int | None
    second_worst: int | None
    least_negative: int | None
    explanation: str
    suggestion: str


def classify_original_rximo(reference_point_min, solution_min, shap_row, target_index,
                            objective_symbols=None, *, tolerance=1e-10):
    """Classify one target; deterministic tie-breaking uses lowest index.

    Mixed/equality situations are resolved using the source's conceptual
    case hierarchy. Ties and zero-only rows are conventions requiring
    comparison against the original authors' implementation.
    """
    rp = np.asarray(reference_point_min, dtype=float).reshape(-1)
    sol = np.asarray(solution_min, dtype=float).reshape(-1)
    phi = np.asarray(shap_row, dtype=float).reshape(-1)
    if rp.size < 2 or sol.size != rp.size or phi.size != rp.size:
        raise ValueError('Reference point, solution and SHAP row must have equal length >= 2')
    if not (np.isfinite(rp).all() and np.isfinite(sol).all() and np.isfinite(phi).all()):
        raise ValueError('All inputs must be finite')
    if not 0 <= target_index < rp.size:
        raise ValueError('Invalid target index')
    names = list(objective_symbols or [f'f_{i+1}' for i in range(rp.size)])
    if len(names) != rp.size:
        raise ValueError('Wrong number of objective names')
    order_worst = sorted(range(phi.size), key=lambda j: (-phi[j], j))
    order_best = sorted(range(phi.size), key=lambda j: (phi[j], j))
    worst = order_worst[0] if phi[order_worst[0]] > tolerance else None
    best = order_best[0] if phi[order_best[0]] < -tolerance else None
    second = next((j for j in order_worst if j != target_index), None)
    external = [j for j in range(phi.size) if j != target_index]
    least_negative = max(external, key=lambda j: (phi[j], -j))
    rival = worst if worst is not None and worst != target_index else (
        second if worst == target_index else least_negative
    )
    # The source groups cases 1-2 when RP is worse than solution in every
    # component, 3-4 when RP is better in every component.
    if np.all(rp > sol + tolerance):
        case = 2 if worst == target_index else 1
    elif np.all(rp < sol - tolerance):
        case = 4 if worst == target_index else 3
    elif best is None:
        case = 5
    elif worst is None:
        case = 6
    elif worst == target_index:
        case = 8
    elif best == target_index:
        case = 9
    else:
        case = 7
    labels = {
        1: 'all aspirations worse than solution',
        2: 'all aspirations worse; target strongest impairer',
        3: 'all aspirations better than solution',
        4: 'all aspirations better; target strongest impairer',
        5: 'no improving SHAP effects',
        6: 'no impairing SHAP effects',
        7: 'mixed improving and impairing effects',
        8: 'target strongest impairer',
        9: 'target strongest improver',
    }
    explanation = (f'Original R-XIMO case {case}: {labels[case]}. '
                   f'The rival for {names[target_index]} is {names[rival]}, selected '
                   'from the minimized-target SHAP row.')
    suggestion = (f'Try improving the aspiration level for {names[target_index]} '
                  f'and relaxing the aspiration level for {names[rival]}.')
    return OriginalRXIMOResult(case, labels[case], target_index, rival,
                               best, worst, second, least_negative,
                               explanation, suggestion)
