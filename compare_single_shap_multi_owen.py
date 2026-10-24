"""Compare single-target Kernel SHAP with multi-target hierarchical Owen explanations.

Run from the MAS-XIMO project root, with paper_scenario_runner.py installed.

Fairness controls: one RPM reference point/solution, one fitted surrogate, one
training dataset, one sampled masking background, and the same target choices.
SHAP explains a *single predicted minimized objective*; Owen explains the
project's *normalized joint achievement-scalarizing utility*. Therefore their
raw attribution magnitudes are NOT directly comparable. This is a workflow
comparison, not an attribution-method ablation or a reproduction of DESDEO2's
native SHAP implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import shap

from paper_scenario_runner import SCENARIOS
from experiment_scenarios import reference_profiles
from problem_setup import create_problem, default_background_path
from explanations.owen_service import OwenExplanationService
from explanations.utils import (
    get_objective_symbols, get_ideal_and_nadir_minimized, orient_objectives_to_minimize,
    orient_objectives_from_minimize, solve_reference_point_with_desdeo_rpm,
)
from explanations.original_rximo_cases import classify_original_rximo
from explanations.counterfactual_validation import (
    build_counterfactual_reference_point, evaluate_counterfactual_solution,
    outcome_to_dict,
)


def single_target_shap(service, reference_point_min, targets, *, background_size, seed, nsamples):
    """Kernel SHAP on the same fitted multi-output GP used by Owen.

    SHAP values explain the *minimized* predicted target value; a negative
    value contributes toward a smaller (better) predicted objective value.
    """
    background = service.data.sample(n=min(background_size, service.data.height), seed=seed)
    background_x = background.select(service.input_symbols).to_numpy().astype(float)
    x = np.asarray(reference_point_min, dtype=float).reshape(1, -1)
    results = {}
    for target in targets:
        symbol = f's_{target}'
        j = service.output_symbols.index(symbol)

        def model(x_in, output_index=j):
            values = service.surrogate.predict(np.atleast_2d(np.asarray(x_in, dtype=float)))
            return np.asarray(values[:, output_index], dtype=float)

        explainer = shap.KernelExplainer(model, background_x)
        values = np.asarray(explainer.shap_values(x, nsamples=nsamples, silent=True), dtype=float).reshape(-1)
        prediction = float(model(x)[0])
        expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        reconstruction_error = abs(expected + float(values.sum()) - prediction)
        results[target] = {
            'target': target,
            'explained_quantity': f'predicted minimized {symbol}',
            'predicted_target_minimized': prediction,
            'expected_value': expected,
            'reconstruction_error': reconstruction_error,
            'shap_values': [
                {'input': name, 'shap_value': float(value)}
                for name, value in zip(service.input_symbols, values, strict=True)
            ],
        }
    return results



def shap_rximo_single_target(problem, service, reference_point_original, reference_point_min,
                             initial_min, target, shap_result, *, step):
    """Original-paper-style single-target SHAP classification, independent of Owen."""
    objective_symbols = get_objective_symbols(problem)
    target_index = objective_symbols.index(target)
    phi = np.array([v['shap_value'] for v in shap_result['shap_values']], dtype=float)
    result = classify_original_rximo(reference_point_min, initial_min, phi,
                                     target_index, objective_symbols)
    actionable_targets = [f'r_{target}']
    selected_rivals = [f'r_{objective_symbols[result.rival_index]}']
    cf_rp = build_counterfactual_reference_point(
        problem, reference_point_original, service.input_symbols,
        actionable_targets, selected_rivals, step_fraction=step,
    )
    cf_min = np.asarray(solve_reference_point_with_desdeo_rpm(problem, cf_rp), dtype=float).reshape(-1)
    cf_original = orient_objectives_from_minimize(problem, cf_min).reshape(-1)
    outcome = outcome_to_dict(evaluate_counterfactual_solution(
        problem, initial_min, cf_min, [target], cf_rp,
    ))
    return {
        'target': target,
        'case_number': result.case_number,
        'case_name': result.case_name,
        'reference_point_status': 'original_rximo',
        'explanation': result.explanation,
        'suggestion': result.suggestion,
        'actionable_target_members': actionable_targets,
        'selected_action_rival_members': selected_rivals,
        'attributions_minimized_target': shap_result['shap_values'],
        'rival_index': result.rival_index,
        'best_effect_index': result.best_effect,
        'worst_effect_index': result.worst_effect,
        'counterfactual_reference_point': np.asarray(cf_rp).reshape(-1).tolist(),
        'counterfactual_solution': cf_original.tolist(),
        'outcome': outcome,
        'note': ('Independent original-paper-style R-XIMO nine-case baseline. '
                 'Minimized-objective SHAP signs retained; ties/zero conventions '
                 'require author-code verification.'),
    }


def run(name, *, seed, samples, background_size, gp_config, hierarchy, nsamples, step, output_dir, target_sets=None):
    spec = SCENARIOS[name]
    problem = create_problem(spec.problem)
    symbols = get_objective_symbols(problem)
    rp = np.asarray(spec.reference_point if spec.reference_point is not None
                    else reference_profiles(problem)[spec.profile], dtype=float)
    target_groups = tuple(tuple(group) for group in target_sets) if target_sets is not None else spec.targets
    if not target_groups or any(not group for group in target_groups):
        raise ValueError("At least one nonempty target set is required")
    unknown = sorted({target for group in target_groups for target in group} - set(symbols))
    if unknown:
        raise ValueError(f"Unknown target objective(s): {unknown}; valid objectives: {symbols}")
    data_path = default_background_path(spec.problem, seed=seed, samples=samples)
    if not data_path.is_file():
        raise FileNotFoundError(f'RPM dataset missing: {data_path}')
    service = OwenExplanationService(
        problem=problem, background_data_path=data_path, surrogate_type='gaussian_process',
        gp_config=gp_config, seed=seed, background_size=background_size, hierarchy=hierarchy,
    )
    initial_min = np.asarray(solve_reference_point_with_desdeo_rpm(problem, rp), dtype=float).reshape(-1)
    initial = orient_objectives_from_minimize(problem, initial_min).reshape(-1)
    rp_min = orient_objectives_to_minimize(problem, rp)
    selected = list(dict.fromkeys(t for group in target_groups for t in group))
    shap_results = single_target_shap(service, rp_min, selected,
                                      background_size=background_size, seed=seed, nsamples=nsamples)
    comparisons = []
    shap_recommendations = {}
    rows = []
    for targets in target_groups:
        explanation = service.explain(reference_point_original=rp, solution_min=initial_min, targets=list(targets))
        cf_rp = build_counterfactual_reference_point(
            problem, rp, service.input_symbols,
            explanation['actionable_target_members'],
            explanation['selected_action_rival_members'], step_fraction=step,
        )
        cf_min = np.asarray(solve_reference_point_with_desdeo_rpm(problem, cf_rp), dtype=float).reshape(-1)
        cf = orient_objectives_from_minimize(problem, cf_min).reshape(-1)
        outcome = outcome_to_dict(evaluate_counterfactual_solution(
            problem, initial_min, cf_min, list(targets), cf_rp,
        ))
        owens = {item['input']: float(item['owen_value']) for item in explanation['owen_values']}
        entry = {
            'targets': list(targets), 'owen_values': explanation['owen_values'],
            'coalition_contributions': explanation['coalition_contributions'],
            'suggestion': explanation['suggestion'], 'explanation': explanation['explanation'],
            'case_number': explanation['case_number'],
            'counterfactual_reference_point': np.asarray(cf_rp).reshape(-1).tolist(),
            'counterfactual_solution': cf.tolist(), 'outcome': outcome,
        }
        comparisons.append(entry)
        row = {
            'scenario': name, 'method': 'hierarchical_owen', 'target_set': '+'.join(targets),
            'explained_quantity': 'normalized joint target utility (project ASF)',
            'suggestion': explanation['suggestion'], 'outcome': outcome.get('outcome_type', ''),
        }
        for i, symbol in enumerate(symbols):
            row[f'attribution_r_{symbol}'] = owens.get(f'r_{symbol}', float('nan'))
            row[f'delta_{symbol}'] = float(cf[i] - initial[i])
        rows.append(row)
    for target, item in shap_results.items():
        recommendation = shap_rximo_single_target(
            problem, service, rp, rp_min, initial_min, target, item, step=step,
        )
        shap_recommendations[target] = recommendation
        values = {v['input']: v['shap_value'] for v in item['shap_values']}
        row = {
            'scenario': name, 'method': 'single_target_kernel_shap_original_rximo', 'target_set': target,
            'explained_quantity': item['explained_quantity'],
            'suggestion': recommendation['suggestion'],
            'outcome': recommendation['outcome'].get('outcome_type', ''),
        }
        for i, symbol in enumerate(symbols):
            row[f'attribution_r_{symbol}'] = values.get(f'r_{symbol}', float('nan'))
            row[f'delta_{symbol}'] = float(
                recommendation['counterfactual_solution'][i] - initial[i]
            )
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_controlled_targets' if target_sets is not None else ''
    prefix = output_dir / f'{name}_seed{seed}_shap_vs_owen{suffix}'
    metadata = {
        'scenario': name, 'description': spec.description, 'problem': spec.problem,
        'target_sets': [list(group) for group in target_groups],
        'seed': seed, 'samples': samples, 'background_size': background_size,
        'gp_config': gp_config, 'hierarchy': hierarchy, 'shap_nsamples': nsamples,
        'counterfactual_step': step, 'dataset': str(data_path),
        'reference_point_original': rp.tolist(), 'initial_solution_original': initial.tolist(),
        'comparison_note': ('Kernel SHAP values explain individual minimized surrogate outputs; '
                            'Owen values explain the normalized joint target utility. '
                            'Do not compare their magnitudes as though they share a scale. '
                            'Both methods have RPM-validated recommendations. Single-target '
                            'SHAP uses a separate original-paper-style nine-case classifier '
                            'on minimized-objective SHAP attributions; '
                            'the single-target SHAP and Owen explained functions differ.'),
    }
    with prefix.with_suffix('.json').open('w', encoding='utf-8') as f:
        json.dump({'configuration': metadata, 'single_target_shap': shap_results,
                   'single_target_shap_rximo_recommendations': shap_recommendations,
                   'owen_explanations': comparisons}, f, indent=2)
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_suffix('.md').open('w', encoding='utf-8') as f:
        f.write(f'# {name}: single-target Kernel SHAP vs hierarchical Owen\n\n')
        f.write(f'Initial reference point: `{rp.tolist()}`  \nInitial RPM solution: `{initial.tolist()}`\n\n')
        f.write('**Interpretation:** SHAP and Owen explain different output quantities; '
                'raw attribution magnitudes must not be compared directly.\n\n')
        f.write('| Method | Target set | ' + ' | '.join(f'r_{s}' for s in symbols) + ' |\n')
        f.write('|---|---|' + '|'.join(['---:'] * len(symbols)) + '|\n')
        for row in rows:
            f.write(f"| {row['method']} | {row['target_set']} | " +
                    ' | '.join(f"{row[f'attribution_r_{s}']:+.6f}" for s in symbols) + ' |\n')
        f.write('\n## Single-target SHAP nine-case recommendations and RPM outcomes\n\n')
        for target, item in shap_recommendations.items():
            f.write(f"### {target} — Case {item['case_number']}: {item['case_name']}\n\n")
            f.write(f"{item['explanation']}\n\n{item['suggestion']}\n\n")
            f.write(f"Counterfactual RP: `{item['counterfactual_reference_point']}`  \n")
            f.write(f"Counterfactual solution: `{item['counterfactual_solution']}`  \n")
            f.write(f"Outcome: `{item['outcome'].get('outcome_type', '')}`\n\n")
        f.write('\n## Owen recommendations and RPM outcomes\n\n')
        for item in comparisons:
            f.write(f"### {', '.join(item['targets'])}\n\n{item['suggestion']}\n\n")
            f.write(f"Counterfactual RP: `{item['counterfactual_reference_point']}`  \n")
            f.write(f"Counterfactual solution: `{item['counterfactual_solution']}`  \n")
            f.write(f"Outcome: `{item['outcome'].get('outcome_type', '')}`\n\n")
    print(f'Wrote {prefix}.json, .csv, .md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=('all', *SCENARIOS), default='river_joint_conflict')
    p.add_argument('--target-set', action='append', nargs='+', metavar='OBJECTIVE',
                   help='Override target groups; repeat for multiple groups, e.g. '
                        '--target-set f_1 --target-set f_3 --target-set f_1 f_3 f_2')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--background-size', type=int, default=100)
    p.add_argument('--gp-config', choices=('current', 'bounded', 'bounded_restarts'), default='bounded_restarts')
    p.add_argument('--hierarchy', choices=('correlation', 'moo-pearson-relationship', 'moo-pearson-conflict',
                                         'moo-spearman-relationship', 'moo-spearman-conflict'), default='moo-spearman-conflict')
    p.add_argument('--shap-nsamples', type=int, default=256)
    p.add_argument('--counterfactual-step', type=float, default=0.10)
    p.add_argument('--output-dir', type=Path, default=Path('results/shap_owen_comparisons'))
    args = p.parse_args()
    if args.shap_nsamples <= 0 or args.background_size <= 0:
        p.error('SHAP samples and background size must be positive')
    if not 0 <= args.counterfactual_step <= 1:
        p.error('Counterfactual step must be between 0 and 1')
    if args.scenario == 'all' and args.target_set is not None:
        p.error('--target-set requires a specific --scenario, not all')
    for name in SCENARIOS if args.scenario == 'all' else (args.scenario,):
        print(f'Running {name}')
        run(name, seed=args.seed, samples=args.samples, background_size=args.background_size,
            gp_config=args.gp_config, hierarchy=args.hierarchy, nsamples=args.shap_nsamples,
            step=args.counterfactual_step, output_dir=args.output_dir,
            target_sets=args.target_set)


if __name__ == '__main__':
    main()
