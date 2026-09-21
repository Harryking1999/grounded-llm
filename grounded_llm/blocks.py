"""Pixel states, strict action/report scoring and paired metrics; no model code."""
from collections import Counter
import copy
import json
from pathlib import Path
import random

from .artifacts import read_json
from experiments.sol_dag_blocks.src.tasks import BlocksTask


def resolve_blocks_config(raw, root):
    keys = ('schema_version', 'task_family', 'study', 'mode', 'output', 'model',
            'adapter', 'conditions', 'blocks', 'training', 'generation', 'smoke', 'evaluation')
    config = {k: copy.deepcopy(raw[k]) for k in keys}
    if raw.get('source_commit'):
        config['source_commit'] = raw['source_commit']
    config['execution'] = copy.deepcopy(raw.get('execution', {'phase': 'all', 'shard_index': 0, 'num_shards': 1}))
    execution = config['execution']
    if not 0 <= execution['shard_index'] < execution['num_shards']:
        raise ValueError('Invalid evaluation shard')
    if execution['phase'] == 'eval' and not execution.get('training_run'):
        raise ValueError('Evaluation requires a saved dataset/training run')
    config['assets'] = {'state_dim': raw['assets']['state_dim']}
    config['training'] = {k: config['training'][k] for k in (
        'enabled', 'epochs', 'report_batch_size', 'initial_microbatch_size', 'shuffle_seed',
        'tasks', 'selection', 'learning_rate', 'betas', 'epsilon', 'weight_decay', 'gradient_clip_norm')}
    config['generation'] = {k: config['generation'][k] for k in (
        'do_sample', 'num_beams', 'attempts_per_item', 'max_new_tokens', 'report_max_new_tokens', 'context_limit')}
    config['evaluation'] = {k: config['evaluation'][k] for k in ('report_probe_count', 'paired_bootstrap_samples', 'report_batch_size')}
    if config['schema_version'] != 1 or config['mode'] != 'run':
        raise ValueError('Blocks requires schema 1 run mode')
    spec = config['blocks']
    for section, key in [('blocks', 'rules_config'), ('model', 'directory'),
                         ('model', 'provenance')]:
        p = Path(config[section][key])
        config[section][key] = str(p if p.is_absolute() else Path(root) / p)
    p = Path(config['output'])
    config['output'] = str(p if p.is_absolute() else Path(root) / p)
    rules = read_json(config['blocks']['rules_config'])['blocks']
    if config['assets']['state_dim'] != rules['grid_size'] ** 2:
        raise ValueError('Adapter input must equal raw board size')
    if spec['state_tokens'] != 1 or spec['target'] != 'empty':
        raise ValueError('First pilot supports one token and empty goals')
    if config['conditions'] != [{'name': 'text', 'adapter': None}, {'name': 'text_token', 'adapter': 'mlp'}]:
        raise ValueError('Pilot requires the fixed paired text/text_token conditions')
    if config['training']['tasks'] != ['report_board']:
        raise ValueError('No action supervision in this study')
    if config['generation']['do_sample'] or config['generation']['attempts_per_item'] != 1:
        raise ValueError('One greedy trajectory per condition')
    if any(n < 1 or n > 8 for n in spec['construction_counts']):
        raise ValueError('This pilot excludes more than eight construction blocks')
    return config


def build_blocks_dataset(config):
    """Split board families first, disallow translated or derived-state leakage."""
    spec = config['blocks']
    task = BlocksTask(read_json(spec['rules_config'])['blocks'])
    rng = random.Random(spec['seed'])
    seen_families, state_owners = set(), {}
    dataset = {name: [] for name in ('train', 'validation', 'test')}
    for split, cases in dataset.items():
        for count in spec['construction_counts']:
            for number in range(spec['families_per_difficulty'][split]):
                for _ in range(10000):
                    mask, construction = 0, []
                    for _ in range(count):
                        boundary = task.boundary(mask)
                        options = [(tile, action) for tile, action in task.placements
                                   if not tile & mask and (not mask or tile & boundary)]
                        if not options:
                            break
                        tile, action = rng.choice(options)
                        mask |= tile
                        construction.append(action)
                    if len(construction) != count or task.normalized_key(mask) in seen_families:
                        continue
                    reports = [mask]
                    for _ in range(spec['reports_per_family'] - 1):
                        retained = rng.sample(construction, rng.randint(1, count))
                        reports.append(sum(task.by_action[a['shape_id'], a['row'], a['col']] for a in retained))
                    keys = {task.normalized_key(m) for m in reports}
                    if any(state_owners.get(k, split) != split for k in keys):
                        continue
                    seen_families.add(task.normalized_key(mask))
                    state_owners.update({k: split for k in keys})
                    cases.append(dict(id=f'{split}_blocks{count}_{number:03}', difficulty=count,
                                      grid='/'.join(task.to_rows(mask)),
                                      report_grids=[task.to_rows(m) for m in reports],
                                      construction_reference=construction))
                    break
                else:
                    raise RuntimeError('Could not construct non-leaking board families')
    return task, dataset


def report_examples(cases):
    return [dict(id=f"{case['id']}_report{i}", rows=rows, task='report_board')
            for case in cases for i, rows in enumerate(case['report_grids'])]


def planning_prompt(task, case, config):
    # Reuse the authoritative baseline rules and shape serialization verbatim.
    rules = task.prompt(case).split('\n\nPlan the COMPLETE sequence')[0]
    return rules + '\n\n' + config['blocks']['protocol']


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


def score_step(task, mask, raw):
    row = dict(accepted=False, action_legal=False, report_correct=False, after=mask)
    try:
        action = strict_json(raw)
        if not isinstance(action, dict):
            raise ValueError('Expected one object')
    except (ValueError, TypeError):
        return dict(row, failure='format_error')
    if action == {'done': True}:
        return dict(row, failure='premature_done' if mask else None, accepted=not mask)
    if set(action) != {'shape_id', 'row', 'col', 'board_after'}:
        return dict(row, failure='format_error')
    try:
        after = task.apply(mask, action)
    except ValueError as error:
        return dict(row, failure='illegal_action', detail=str(error))
    row.update(action_legal=True, after=after)
    try:
        report = action['board_after']
        if not isinstance(report, list) or task.from_grid(report) != after:
            raise ValueError('Incorrect state')
    except (ValueError, TypeError):
        return dict(row, failure='state_report_error')
    return dict(row, accepted=True, report_correct=True, failure=None)


def summarize_blocks(rows, config):
    import numpy as np
    result = {'episodes': len(rows), 'groups': {}, 'paired': {}}
    for difficulty in config['blocks']['construction_counts']:
        for condition in ('text', 'text_token'):
            values = [r for r in rows if r['condition'] == condition and r['difficulty'] == difficulty]
            if not values:
                continue
            result['groups'][f'{condition}/blocks{difficulty}'] = dict(
                solved=sum(r['solved'] for r in values), total=len(values),
                failures=dict(Counter(r['failure'] for r in values if r['failure'])),
                attempted_steps=sum(len(r['steps']) for r in values),
                legal_actions=sum(r['legal_actions'] for r in values),
                accepted_steps=sum(r['accepted_steps'] for r in values),
                correct_reports=sum(s['report_correct'] for r in values for s in r['steps']),
                generated_tokens=sum(r['generated_tokens'] for r in values),
                prefill_tokens=sum(r['prefill_tokens'] for r in values),
                seconds=sum(r['seconds'] for r in values))
        paired = {r['id']: {} for r in rows if r['difficulty'] == difficulty}
        for r in rows:
            if r['difficulty'] == difficulty:
                paired[r['id']][r['condition']] = r
        complete = [p for p in paired.values() if set(p) == {'text', 'text_token'}]
        if not complete:
            continue
        rng = np.random.default_rng(config['blocks']['seed'])
        changes = {}
        for metric in ('solved', 'legal_actions', 'accepted_steps'):
            diff = np.array([int(p['text_token'][metric]) - int(p['text'][metric]) for p in complete])
            bootstrap = diff[rng.integers(0, len(diff), (config['evaluation']['paired_bootstrap_samples'], len(diff)))].mean(1)
            changes[metric] = dict(mean_difference=float(diff.mean()),
                                   descriptive_95_percent_interval=np.quantile(bootstrap, [.025, .975]).tolist())
        result['paired'][f'blocks{difficulty}'] = dict(n=len(complete), **changes)
    return result
