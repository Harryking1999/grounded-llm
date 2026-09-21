"""Pixel states, strict action/report scoring and paired metrics; no model code."""
from collections import Counter
import copy
import json
from pathlib import Path
import random

from .artifacts import read_json, read_jsonl, write_json
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
    if execution['phase'] in ('fit', 'eval', 'summarize') and not execution.get('training_run'):
        raise ValueError('Evaluation requires a saved dataset/training run')
    config['assets'] = {'state_dim': raw['assets']['state_dim']}
    config['training'] = {k: config['training'][k] for k in (
        'enabled', 'epochs', 'report_batch_size', 'initial_microbatch_size', 'shuffle_seed',
        'tasks', 'selection', 'learning_rate', 'betas', 'epsilon', 'weight_decay', 'gradient_clip_norm')}
    config['training']['validation_interval'] = raw['training'].get('validation_interval', 1)
    config['readout_requirement'] = copy.deepcopy(raw.get('readout_requirement', {'minimum_exact_accuracy': 0.95}))
    if raw.get('fit_diagnostic'):
        config['fit_diagnostic'] = copy.deepcopy(raw['fit_diagnostic'])
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


def require_readable_adapter(training_run, config):
    """Training completion alone never authorizes planning with an adapter."""
    saved_config = Path(training_run) / 'config.json'
    if saved_config.exists() and read_json(saved_config).get('execution', {}).get('phase') == 'fit':
        raise ValueError('Small-set fit is not independent validation and cannot authorize planning')
    path = Path(training_run) / 'adapter/training_summary.json'
    summary = read_json(path)
    accuracy = summary.get('selected_validation_accuracy')
    minimum = config['readout_requirement']['minimum_exact_accuracy']
    if accuracy is None or accuracy < minimum:
        raise ValueError(f'Adapter not ready for planning: validation exact accuracy {accuracy}; required {minimum}')


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


def collect_blocks_runs(training_run, episode_runs, output):
    """Combine completed shards once, rejecting omissions and repeated episodes."""
    training_run = Path(training_run)
    config = read_json(training_run / 'config.json')
    dataset = read_json(training_run / 'dataset.json')
    expected = {(case['id'], condition['name']) for case in dataset['test'] for condition in config['conditions']}
    seen, rows = set(), []
    for directory in map(Path, episode_runs):
        saved = read_json(directory / 'config.json')
        for key in ('blocks', 'model', 'adapter', 'generation', 'training'):
            if saved[key] != config[key]:
                raise ValueError(f'Shard contract differs: {directory}, {key}')
        for row in read_jsonl(directory / 'episodes.jsonl'):
            key = row['id'], row['condition']
            if key not in expected or key in seen:
                raise ValueError(f'Unexpected or repeated episode: {key}')
            seen.add(key)
            rows.append(row)
    if seen != expected:
        raise ValueError(f'Incomplete paired evaluation: {len(seen)} / {len(expected)}')
    result = summarize_blocks(rows, config)
    result.update(readout=read_json(training_run / 'readout.json'),
                  training=read_json(training_run / 'adapter/training_summary.json'),
                  training_cost=read_json(training_run / 'training_cost.json'),
                  source_runs=[str(p) for p in episode_runs], training_run=str(training_run))
    reports_path = training_run / 'report_predictions.jsonl'
    if reports_path.exists():
        result['readout_diagnostics'] = report_diagnostics(read_jsonl(reports_path))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', result)
    write_blocks_report(output / 'report.md', result)
    return result


def report_diagnostics(rows):
    """Separate occupied-cell recall from majority-empty-cell accuracy."""
    groups = {}
    for row in rows:
        group = groups.setdefault(str(row['epoch']), Counter())
        group['total'] += 1
        expected = ''.join(row['expected'])
        group['true_occupied'] += expected.count('1')
        group['all_zero_baseline_cell_matches'] += expected.count('0')
        group['total_cells'] += len(expected)
        try:
            value = strict_json(row['raw_output'])
            if not isinstance(value, list) or len(value) != len(row['expected']):
                raise ValueError('Bad report shape')
            if any(not isinstance(r, str) or len(r) != len(truth) or set(r) - {'0', '1'}
                   for r, truth in zip(value, row['expected'])):
                raise ValueError('Bad board rows')
        except (TypeError, ValueError):
            group['invalid'] += 1
            continue
        predicted = ''.join(value)
        group['valid'] += 1
        group['exact'] += predicted == expected
        group['all_zero_outputs'] += '1' not in predicted
        group['cell_matches'] += sum(a == b for a, b in zip(predicted, expected))
        group['predicted_occupied'] += predicted.count('1')
        group['true_positive_occupied'] += sum(a == b == '1' for a, b in zip(predicted, expected))
    return {key: dict(value) for key, value in groups.items()}


def write_blocks_report(path, summary):
    lines = ['# Blocks Step 2 首轮结果', '',
             '相同文字输入与文字加状态 token；动作或动作后自报错误均立即失败。', '',
             '| 构造块数 | 文字成功 | +token 成功 | 文字合法动作 | +token 合法动作 | 文字通过步骤 | +token 通过步骤 |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    difficulties = sorted(int(k.split('blocks')[-1]) for k in summary['groups'] if k.startswith('text/'))
    for count in difficulties:
        text, token = (summary['groups'][f'{condition}/blocks{count}'] for condition in ('text', 'text_token'))
        lines.append('| ' + ' | '.join(map(str, [count, f"{text['solved']}/{text['total']}",
            f"{token['solved']}/{token['total']}", text['legal_actions'], token['legal_actions'],
            text['accepted_steps'], token['accepted_steps']])) + ' |')
    readout = summary['readout']
    lines += ['', f"选中 adapter 独立棋盘读出：{readout['correct']}/{readout['total']}。",
              '合法动作但自报错误仍计入合法动作数；通过步骤要求二者均正确。', '',
              '## 解释边界', '',
              '单训练 seed、每题一条 greedy 轨迹；配对区间见 summary.json，不能表示训练随机性。',
              '若棋盘读出失败，本轮只能评价这次训练得到的接口，不能证伪准确状态 token 的规划价值。',
              '报告 loss 下降不等于状态可读；占据格召回和全零输出统计见读出诊断。', '',
              '## 运行来源', '', f"训练：`{summary['training_run']}`。",
              '评测分片、实际成本、全部失败类型与配对差值见 summary.json。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
