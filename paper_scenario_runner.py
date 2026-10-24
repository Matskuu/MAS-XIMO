"""Reproduce fixed single-vs-multi-target MAS-XIMO decision scenarios.

Uses the same OwenExplanationService, R-XIMO suggestion and DESDEO RPM
counterfactual functions as the interactive system. No algorithms are copied.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment_scenarios import reference_profiles
from evaluate_owen_explanations import build_utility_explainer, target_utility_from_outputs
from explanations.counterfactual_validation import (
    build_counterfactual_reference_point, evaluate_counterfactual_solution, outcome_to_dict,
)
from explanations.owen_service import OwenExplanationService
from explanations.utils import (
    get_objective_symbols, orient_objectives_from_minimize, solve_reference_point_with_desdeo_rpm,
)
from problem_setup import create_problem, default_background_path


@dataclass(frozen=True)
class Scenario:
    problem: str
    profile: str | None
    reference_point: tuple[float, ...] | None
    targets: tuple[tuple[str, ...], ...]
    description: str


SCENARIOS = {
    'river_joint_conflict': Scenario(
        'river', None, (5.5, 3.0, 5.0, -4.0),
        (('f_1',), ('f_3',), ('f_1', 'f_3')),
        'Same initial solution: separate city oxygen and fishery ROI explanations versus a joint explanation.',
    ),
    'river_three_targets': Scenario(
        'river', 'near_ideal', None,
        (('f_1',), ('f_2',), ('f_3',), ('f_1', 'f_2', 'f_3')),
        'Three-target aggregate utility versus individual target outcomes.',
    ),
    'forest_expanding_targets': Scenario(
        'forest', 'balanced', None,
        (('Rev',), ('Carb',), ('Rev', 'Carb'), ('Rev', 'HA', 'Carb')),
        'Revenue-carbon compromise and adding habitat availability.',
    ),
    'dtlz2_compatible': Scenario(
        'dtlz2', 'near_ideal', None,
        (('f_1',), ('f_2',), ('f_1', 'f_2')),
        'Single and joint explanations for compatible target improvements.',
    ),
}


def run_scenario(name: str, *, seed: int, samples: int, background_size: int,
                 gp_config: str, hierarchy: str, step: float, output_dir: Path) -> None:
    spec = SCENARIOS[name]
    problem = create_problem(spec.problem)
    symbols = get_objective_symbols(problem)
    rp = np.asarray(spec.reference_point if spec.reference_point is not None
                    else reference_profiles(problem)[spec.profile], dtype=float)
    if rp.shape != (len(symbols),):
        raise ValueError(f'{name}: reference point dimension mismatch')
    path = default_background_path(spec.problem, seed=seed, samples=samples)
    if not path.is_file():
        raise FileNotFoundError(f'Missing dataset: {path}. Generate it first; no silent regeneration.')

    service = OwenExplanationService(
        problem=problem, background_data_path=path, surrogate_type='gaussian_process',
        gp_config=gp_config, seed=seed, background_size=background_size,
        hierarchy=hierarchy,
    )
    original_min = solve_reference_point_with_desdeo_rpm(problem, rp)
    original = orient_objectives_from_minimize(problem, original_min).reshape(-1)
    rows = []
    details = []
    for targets in spec.targets:
        explanation = service.explain(reference_point_original=rp, solution_min=original_min,
                                      targets=list(targets))
        adjusted_rp = build_counterfactual_reference_point(
            problem, rp, service.input_symbols,
            explanation['actionable_target_members'],
            explanation['selected_action_rival_members'], step_fraction=step,
        )
        adjusted_min = solve_reference_point_with_desdeo_rpm(problem, adjusted_rp)
        adjusted = orient_objectives_from_minimize(problem, adjusted_min).reshape(-1)
        outcome = evaluate_counterfactual_solution(problem, original_min, adjusted_min,
                                                   list(targets), adjusted_rp)
        outcome_dict = outcome_to_dict(outcome)
        utility = build_utility_explainer(service, list(targets))
        utility_before = float(target_utility_from_outputs(utility, original_min)[0])
        utility_after = float(target_utility_from_outputs(utility, adjusted_min)[0])
        owens = {item['symbol']: float(item['owen_value'])
                 for item in explanation['owen_values']}
        row = {
            'scenario': name, 'problem': spec.problem, 'seed': seed,
            'target_set': '+'.join(targets), 'n_targets': len(targets),
            'case_number': explanation['case_number'],
            'outcome': outcome.outcome_type,
            'joint_target_improvement': outcome_dict['joint_target_improvement'],
            'target_utility_change': utility_after - utility_before,
            'selected_rivals': json.dumps(explanation['selected_action_rival_members']),
            'actionable_targets': json.dumps(explanation['actionable_target_members']),
            'suggestion': explanation['suggestion'],
        }
        for i, symbol in enumerate(symbols):
            row[f'r_{symbol}'] = float(rp[i])
            row[f's_{symbol}'] = float(original[i])
            row[f'cf_r_{symbol}'] = float(adjusted_rp[i])
            row[f'cf_s_{symbol}'] = float(adjusted[i])
            sign = 1.0 if bool(getattr(problem.objectives[i], 'maximize', False)) else -1.0
            row[f'improvement_{symbol}'] = float((adjusted[i] - original[i]) * sign)
            row[f'owen_r_{symbol}'] = owens.get(f'r_{symbol}', float('nan'))
        rows.append(row)
        details.append({
            'scenario': name, 'description': spec.description, 'targets': list(targets),
            'reference_point': rp.tolist(), 'initial_solution': original.tolist(),
            'counterfactual_reference_point': adjusted_rp.tolist(),
            'counterfactual_solution': adjusted.tolist(),
            'owen_values': explanation['owen_values'],
            'coalition_contributions': explanation['coalition_contributions'],
            'explanation': explanation['explanation'],
            'suggestion': explanation['suggestion'],
            'case_number': explanation['case_number'],
            'reference_point_status': explanation['reference_point_status'],
            'selected_rivals': explanation['selected_action_rival_members'],
            'actionable_targets': explanation['actionable_target_members'],
            'outcome': outcome_dict,
            'utility_before': utility_before, 'utility_after': utility_after,
        })
        print(f'  {"+".join(targets):20s} {outcome.outcome_type:20s} '
              f'utility delta={utility_after-utility_before:+.6g}')

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f'{name}_seed{seed}'
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix('.json').open('w', encoding='utf-8') as file:
        json.dump({'configuration': {'scenario': name, 'seed': seed, 'samples': samples,
                   'background_size': background_size, 'gp_config': gp_config,
                   'hierarchy': hierarchy, 'counterfactual_step': step,
                   'dataset': str(path)}, 'results': details}, file, indent=2)
    with prefix.with_suffix('.md').open('w', encoding='utf-8') as file:
        file.write(f'# {name}\n\n{spec.description}\n\n')
        file.write(f'Problem: {spec.problem}; seed: {seed}; reference point: {rp.tolist()}\n\n')
        file.write('| Component | ' + ' | '.join('+'.join(t) for t in spec.targets) + ' |\n')
        file.write('|---|' + '|'.join(['---:'] * len(rows)) + '|\n')
        for symbol in symbols:
            file.write(f'| r_{symbol} | ' + ' | '.join(
                f'{row[f"owen_r_{symbol}"]:+.6f}' for row in rows) + ' |\n')
        file.write('\n| Metric | ' + ' | '.join('+'.join(t) for t in spec.targets) + ' |\n')
        file.write('|---|' + '|'.join(['---:'] * len(rows)) + '|\n')
        for symbol in symbols:
            file.write(f'| Improvement {symbol} | ' + ' | '.join(
                f'{row[f"improvement_{symbol}"]:+.6f}' for row in rows) + ' |\n')
        file.write('| Target utility change | ' + ' | '.join(
            f'{row["target_utility_change"]:+.6f}' for row in rows) + ' |\n')
        file.write('| Outcome | ' + ' | '.join(row['outcome'] for row in rows) + ' |\n')
        file.write('\nNote: Owen values are baseline-relative attributions, not directional effects. '
                   'Improvement is positive in the preferred objective direction.\n')
    print(f'  Saved {prefix}.csv, .json, .md')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=('all', *SCENARIOS), default='river_joint_conflict')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--background-size', type=int, default=100)
    parser.add_argument('--gp-config', choices=('current', 'bounded', 'bounded_restarts'),
                        default='bounded_restarts')
    parser.add_argument('--hierarchy', choices=('correlation', 'moo-pearson-relationship',
                        'moo-pearson-conflict', 'moo-spearman-relationship',
                        'moo-spearman-conflict'), default='moo-spearman-conflict')
    parser.add_argument('--counterfactual-step', type=float, default=0.10)
    parser.add_argument('--output-dir', type=Path, default=Path('results/paper_scenarios'))
    args = parser.parse_args()
    if not 0 <= args.counterfactual_step <= 1:
        parser.error('--counterfactual-step must be between 0 and 1')
    selected = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    for name in selected:
        print(f'\nScenario: {name}')
        run_scenario(name, seed=args.seed, samples=args.samples,
                     background_size=args.background_size, gp_config=args.gp_config,
                     hierarchy=args.hierarchy, step=args.counterfactual_step,
                     output_dir=args.output_dir)


if __name__ == '__main__':
    main()
