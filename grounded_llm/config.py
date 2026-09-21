"""Resolve inherited configs and validate the executable harness contract."""
import copy
from pathlib import Path
import yaml
from .artifacts import ROOT
from .prompts import INPUT_KEYS

TASK_ALIASES = {'state_report': ['report_current', 'report_goal'], 'one_step_action': ['action']}

def tasks(value):
    value = [value] if isinstance(value, str) else value
    return [task for name in value for task in TASK_ALIASES.get(name, [name])]

def merge(base, changed):
    result = copy.deepcopy(base)
    for key, value in changed.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result

def load_config(path, root=ROOT, _seen=None):
    """JSON is accepted as YAML. Relative bindings are repository-relative."""
    path = Path(path)
    path = path if path.is_absolute() else Path(root) / path
    path = path.resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError('Config inheritance cycle')
    seen.add(path)
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Config must be a mapping')
    base = load_config(config['base_config'], root, seen) if config.get('base_config') else {}
    result = merge(base, config)
    for stage in ('train', 'eval'):
        spec = result.get(stage, {})
        if spec.get('prompt_config'):
            wording = load_config(spec['prompt_config'], root, seen)
            spec['prompts'] = wording['prompts']
            spec['generation'] = wording['generation']
    if result.get('train', {}).get('data_config'):
        data_contract = load_config(result['train']['data_config'], root, seen)
        result['train'] = merge(data_contract['training'], result['train'])
        result['train']['data_spec'] = {k: data_contract[k] for k in ('action_data', 'report_replay')}
    return result

def input_config(value):
    if set(value) != set(INPUT_KEYS) or any(type(v) is not bool for v in value.values()):
        raise ValueError(f'Input requires exactly five boolean flags: {INPUT_KEYS}')
    return value

def resolve_config(raw, root=ROOT):
    config = copy.deepcopy(raw)
    if config.get('task_family') == 'blocks':
        from .blocks import resolve_blocks_config
        return resolve_blocks_config(config, root)
    if config.get('schema_version') != 1:
        raise ValueError('Use a schema_version: 1 harness config; migration examples are in configs/harness/')
    if config.get('mode') not in ('rescore', 'run'):
        raise ValueError('mode must explicitly be rescore or run')
    config['input'] = input_config(config['input'])
    config['train_input'] = input_config(config.get('train_input', config['input']))
    config['eval_input'] = input_config(config.get('eval_input', config['input']))
    train = merge(config.get('training', {}), config.get('train', {}))
    train['enabled'] = train.get('enabled', False)
    if type(train['enabled']) is not bool:
        raise ValueError('train.enabled must be boolean')
    train['tasks'] = tasks(train.pop('task', train.get('tasks', ['report_current', 'report_goal'])))
    train.setdefault('shuffle_seed', config['data'].get('report_shuffle_seed', 18))
    train.setdefault('selection', 'validation')
    if train['selection'] not in ('validation', 'final'):
        raise ValueError('Checkpoint selection must be validation or final')
    config['training'] = train
    evaluation = merge({'split': 'test', 'tasks': ['report_current', 'report_goal', 'action'],
                        'templates': config.get('prompts', {}).get('test_templates', ['canonical']),
                        'batch_size': 8, 'decoding': 'free'}, config.get('eval', {}))
    evaluation['tasks'] = tasks(evaluation.pop('task', evaluation['tasks']))
    if evaluation['split'] not in ('train', 'validation', 'test'):
        raise ValueError('Unknown evaluation split')
    if evaluation['decoding'] not in ('free', 'node_id_grammar', 'choice_grammar'):
        raise ValueError('Unknown decoding contract')
    config['eval'] = evaluation
    scorer = config['scoring']['type']
    if scorer == 'one_step_success':
        config['scoring']['type'] = scorer = 'strict_node'
    if scorer not in ('strict_node', 'semantic_action', 'choice'):
        raise ValueError('Unknown scorer')
    diagnostics = config.get('diagnostics', [])
    if set(diagnostics) - {'template_consistency', 'swap', 'geometry'}:
        raise ValueError('Unknown diagnostic')
    if set(diagnostics) & {'template_consistency', 'swap'} and scorer != 'strict_node':
        raise ValueError('Historical template/swap diagnostics require the strict node scorer')
    if 'template_consistency' in diagnostics and not {'canonical', 'heldout'} <= set(evaluation['templates']):
        raise ValueError('Template consistency requires canonical and heldout predictions')
    if not evaluation['tasks'] or not evaluation['templates'] or len(evaluation['tasks']) != len(set(evaluation['tasks'])):
        raise ValueError('Evaluation tasks/templates must be nonempty and tasks unique')
    if scorer != 'choice' and set(train['tasks'] + evaluation['tasks']) - {'report_current', 'report_goal', 'action'}:
        raise ValueError('Unknown graph task')
    if scorer == 'semantic_action' and evaluation['tasks'] != ['action']:
        raise ValueError('Semantic action metrics require action-only evaluation')
    if scorer == 'choice' and train['enabled']:
        raise ValueError('Existing decision probes are evaluation-only')
    if scorer == 'choice' and (set(evaluation['tasks']) - {'neighbor_direct', 'neighbor_reason', 'distance_compare', 'transition_compare'} or evaluation['decoding'] != 'choice_grammar'):
        raise ValueError('Existing decision probes require known tasks and choice_grammar')
    if scorer != 'choice' and evaluation['decoding'] == 'choice_grammar':
        raise ValueError('choice_grammar requires decision probe tasks')
    if 'conditions' not in config or raw.get('interface', {}).get('type'):
        kind = config['interface']['type']
        config['conditions'] = [dict(name=kind, adapter=None if kind == 'none' else kind,
                                    checkpoint=config['interface'].get('checkpoint'))]
    names = [c['name'] for c in config['conditions']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Conditions must have unique names')
    for condition in config['conditions']:
        if condition['adapter'] not in (None, 'linear', 'mlp'):
            raise ValueError('Unsupported adapter')
        if Path(condition['name']).name != condition['name'] or condition['name'] in ('.', '..'):
            raise ValueError('Condition name must be a simple directory name')
        for stage in ('train', 'eval'):
            flags = input_config(condition.get(stage + '_input', config[stage + '_input']))
            if condition['adapter'] is None and (flags['current_vector'] or flags['goal_vector']):
                raise ValueError('Vector input requires an adapter')
        if train['enabled'] and condition.get('train', condition['adapter'] is not None):
            if condition['adapter'] is None:
                raise ValueError('Only adapter conditions can train')
            if train.get('cache_frozen_prefix') and config.get('prompt_style', 'state') != 'state':
                raise ValueError('Historical prefix caching supports state prompts only')
            if train.get('cache_frozen_prefix') and not condition.get('train_input', config['train_input'])['current_vector']:
                raise ValueError('Historical prefix caching requires the current state slot')
        if scorer == 'choice':
            name = condition['decision_condition']
            vectors = name != 'text'
            text = not name.endswith('_latent')
            expected = dict(graph_text=text, current_id_text=text, goal_id_text=text,
                            current_vector=vectors, goal_vector=vectors)
            if condition.get('eval_input', config['eval_input']) != expected:
                raise ValueError('Decision probe visibility must agree with decision_condition')
    for generation in (config['generation'], merge(config['generation'], evaluation.get('generation', {})), merge(config['generation'], train.get('generation', {}))):
        if generation.get('do_sample', False) or generation.get('num_beams', 1) != 1 or generation.get('attempts_per_item', 1) != 1:
            raise ValueError('Existing harness supports one greedy answer per item')
        if type(generation['max_new_tokens']) is not int or generation['max_new_tokens'] < 1 or evaluation['batch_size'] < 1:
            raise ValueError('Generation and batch budgets must be positive')
    if train['enabled']:
        if not train['tasks'] or len(train['tasks']) != len(set(train['tasks'])):
            raise ValueError('Training tasks must be nonempty and unique')
        for key in ('epochs', 'report_batch_size', 'initial_microbatch_size'):
            if type(train[key]) is not int or train[key] < 1:
                raise ValueError(f'training.{key} must be positive')
        if train.get('optimizer', 'AdamW') != 'AdamW' or train.get('schedule', 'constant') != 'constant':
            raise ValueError('Existing training supports AdamW with a constant learning rate')
    if config['mode'] == 'rescore' and train['enabled']:
        raise ValueError('Rescore cannot enable training')
    # Resolve only file bindings, leaving model IDs, wording and source revisions intact.
    def absolute(value):
        p = Path(value)
        return str((p if p.is_absolute() else Path(root) / p).resolve())
    for section, keys in {'data': ('asset_dir', 'graph_path', 'split_path', 'items_path', 'permutation_path'),
                          'model': ('directory', 'provenance'), 'assets': ('graph_reference',),
                          'scoring': ('adjudications', 'reference_metrics'),
                          'eval': ('items_path',)}.items():
        for key in keys:
            if config.get(section, {}).get(key):
                config[section][key] = absolute(config[section][key])
    config['output'] = absolute(config['output'])
    if config.get('predictions'):
        paths = config['predictions']
        config['predictions'] = [absolute(p) for p in ([paths] if isinstance(paths, str) else paths)]
    for condition in config['conditions']:
        for key in ('checkpoint', 'initial_checkpoint'):
            if condition.get(key):
                condition[key] = absolute(condition[key])
        if condition.get('checkpoints'):
            condition['checkpoints'] = {name: absolute(path) if path else None for name, path in condition['checkpoints'].items()}
    return config
