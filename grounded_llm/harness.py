"""One config-driven entry for graph interfaces, training, evaluation and replay.

rescore never imports torch/transformers or loads a model. It validates every
stored verdict before writing a new run, and stops on the first disagreement.
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import subprocess
import shutil
import yaml

from .artifacts import ROOT, append_json, chunks, read_json, read_jsonl, write_json
from .config import load_config, resolve_config
from .data import build_dataset, examples, training_examples
from .prompts import adjacency_text
from .scoring import ScoringMismatch, diagnostics, metrics, score_record

def predict(interface, adapter, items, batch_size, adj, distances, condition, checkpoint, path=None, extra=None):
    """Historical predict API delegates generation and scoring to their owners."""
    rows = []
    for batch in chunks(items, batch_size):
        for item, generation in zip(batch, interface.generate(batch, adapter)):
            raw = dict(item, **generation, condition=condition, checkpoint=checkpoint, **(extra or {}))
            row = score_record(raw, adj, distances, 'strict_node')
            rows.append(row)
            if path is not None:
                append_json(path, row)
    return rows

def stage_config(config, condition, stage):
    changed = copy.deepcopy(config)
    changed['input'] = condition.get(stage + '_input', config[stage + '_input'])
    spec = config['training'] if stage == 'train' else config['eval']
    if 'prompts' in spec:
        changed['prompts'] = spec['prompts']
    if 'generation' in spec:
        changed['generation'].update(spec['generation'])
    changed['prompt_style'] = spec.get('prompt_style', config.get('prompt_style', 'state'))
    return changed

def contract(config, dataset, rows=None):
    """Describe actual inputs/targets, saved split, checkpoints and judge."""
    evaluation = config['eval']
    checkpoints = [{k: r[k] for k in ('condition', 'checkpoint') if k in r} for r in (rows or [])]
    distinct = {tuple(sorted(c.items())) for c in checkpoints}
    actual_checkpoints = []
    for entry in sorted(distinct):
        identity = dict(entry)
        condition = next((c for c in config['conditions'] if c['name'] == identity.get('condition')), {})
        bound = condition.get('checkpoints', {}).get(identity.get('checkpoint')) or condition.get('checkpoint')
        if bound:
            identity['path'] = bound
        elif identity.get('checkpoint') in ('initial', 'selected', 'final'):
            identity['source_prediction_files'] = config.get('predictions', [])
            identity['source_checkpoint_filename'] = identity['checkpoint'] + '.safetensors'
        actual_checkpoints.append(identity)
    return dict(schema_version=1, study=config.get('study'), mode=config['mode'],
        dataset=dict(graph=config['data'].get('graph_path', config['assets'].get('graph_reference')),
                     asset_dir=config['data'].get('asset_dir'), map_file=config['assets'].get('map_file'),
                     node_count=len(dataset.adjacency), split_path=str(dataset.split_path),
                     split_counts={name: len(pairs) for name, pairs in dataset.split.items()},
                     pair_unit='unordered pair; both directions stay in the same split'),
        training=dict(enabled=config['training']['enabled'], split='train', validation_split='validation',
                      tasks=config['training']['tasks'],
                      input={c['name']: stage_config(config, c, 'train')['input'] for c in config['conditions']},
                      prompts=config['training'].get('prompts', config.get('prompts')),
                      template='canonical',
                      target='correct node ID tokens followed by native EOS; prompt/padding masked',
                      action_target='one seeded uniformly chosen successful neighbor when action is enabled',
                      data_spec=config['training'].get('data_spec'),
                      pools={c['name']: c.get('pool', 'configured_tasks') for c in config['conditions']},
                      loss='mean per-answer token cross entropy, then equal mean over samples',
                      trainable='adapter only; LLM/Q/V frozen',
                      selection=config['training']['selection'],
                      conditions={c['name']: c.get('train', c['adapter'] is not None) for c in config['conditions']}),
        evaluation=dict(input={c['name']: stage_config(config, c, 'eval')['input'] for c in config['conditions']},
                        tasks=evaluation['tasks'], split=evaluation['split'], templates=evaluation['templates'],
                        items_path=evaluation.get('items_path', config['data'].get('items_path')),
                        state_roles=['a', 'b', 'g'] if config['scoring']['type'] == 'choice' else ['u', 'g'],
                        permutation_path=config['data'].get('permutation_path'),
                        checkpoints=actual_checkpoints if rows else
                            {c['name']: c.get('checkpoints', c.get('checkpoint', 'frozen' if c['adapter'] is None else None)) for c in config['conditions']},
                        checkpoint_bindings={c['name']: {k: c[k] for k in ('adapter', 'checkpoint', 'initial_checkpoint', 'checkpoints', 'decision_condition') if k in c} for c in config['conditions']},
                        generation=stage_config(config, config['conditions'][0], 'eval')['generation'],
                        prompt_style=evaluation.get('prompt_style', 'state'),
                        prompts=evaluation.get('prompts', config.get('prompts')), decoding=evaluation['decoding'],
                        diagnostics=config.get('diagnostics', [])),
        scorer=dict(type=config['scoring']['type'], implementation='grounded_llm.scoring',
                    successful_action='adjacency[u,v] and distance[v,g] < distance[u,g]',
                    denominator='all predictions in each condition/checkpoint/template/decoding/split group',
                    adjudications=config['scoring'].get('adjudications'),
                    verify_existing=config['scoring'].get('verify_existing', True)),
        prediction_sources=config.get('predictions', []))

def provenance(config):
    versions = {}
    for package in ('numpy', 'PyYAML', 'torch', 'transformers', 'tokenizers', 'safetensors'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    archived = config.get('source_commit')
    record = dict(source_commit=archived or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_status='committed archive' if archived else subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).strip(),
                source_root=str(ROOT), python=platform.python_version(), versions=versions,
                started_utc=datetime.now(timezone.utc).isoformat(),
                mode=config['mode'], prediction_sources=config.get('predictions', []),
                model=config['model'], assets=config['assets'])
    if config['model'].get('provenance'):
        record['model_provenance'] = read_json(config['model']['provenance'])
    record['prediction_source_runs'] = []
    for source in config.get('predictions', []):
        parent = Path(source).parent
        runtime = next((p for p in (parent / 'runtime.json', parent.parent / 'runtime.json') if p.exists()), None)
        if runtime:
            record['prediction_source_runs'].append(dict(predictions=source, runtime_path=str(runtime), runtime=read_json(runtime)))
    return record

def start_run(config, dataset):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    write_json(output / 'contract.json', contract(config, dataset))
    write_json(output / 'split.json', dataset.split)
    write_json(output / 'provenance.json', provenance(config))
    for name in ('predictions.jsonl', 'scored_predictions.jsonl'):
        (output / name).touch()
    return output

def finish_run(config, dataset, output, rows, result):
    write_json(output / 'contract.json', contract(config, dataset, rows))
    write_json(output / 'metrics.json', result)
    record = read_json(output / 'provenance.json')
    record['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'provenance.json', record)

def reviews_for(config):
    path = config['scoring'].get('adjudications')
    result = {}
    for row in read_json(path) if path else []:
        key = (row['condition'], row['u'], row['g'])
        if key in result or not row.get('evidence') or not row.get('rationale'):
            raise ValueError('Semantic review requires unique key, evidence and rationale')
        result[key] = dict(node=row['node'], source='semantic_review',
                           status='identified' if row['node'] is not None else row['status'])
    return result

def rescore(config, dataset):
    sources = config.get('predictions')
    if not sources:
        raise ValueError('Rescore requires existing predictions paths')
    originals, scored = [], []
    reviews = reviews_for(config)
    seen = set()
    for source in sources:
        for row in read_jsonl(source):
            if row['task'] not in config['eval']['tasks'] or row.get('split', config['eval']['split']) != config['eval']['split']:
                raise ValueError('Prediction task/split disagrees with replay contract')
            if row.get('condition') not in {c['name'] for c in config['conditions']}:
                raise ValueError('Prediction condition disagrees with replay contract')
            if config['scoring']['type'] != 'choice':
                if row['template'] not in config['eval']['templates']:
                    raise ValueError('Prediction template disagrees with replay contract')
                if tuple(sorted((row['u'], row['g']))) not in set(map(tuple, dataset.split[config['eval']['split']])):
                    raise ValueError('Prediction pair is outside the configured saved split')
            key = tuple((k, row[k]) for k in ('condition', 'checkpoint', 'template', 'decoding', 'split', 'variant', 'task', 'u', 'g', 'id') if k in row)
            if key in seen:
                raise ValueError('Repeated prediction identity; check selected sources')
            seen.add(key)
            decision = reviews.get((row.get('condition'), row.get('u'), row.get('g')))
            new = score_record(row, dataset.adjacency, dataset.distances, config['scoring']['type'],
                               config['scoring'].get('verify_existing', True), decision)
            originals.append(row)
            scored.append(new)
    if not originals:
        raise ValueError('Predictions file is empty')
    result = metrics(scored, config['scoring']['type'])
    if config['scoring'].get('verify_existing', True):
        # Compare full metric payloads as well as individual verdicts.
        if result != metrics(originals, config['scoring']['type']):
            raise ScoringMismatch('Stored and recomputed aggregate metrics differ')
    reference = config['scoring'].get('reference_metrics')
    if reference and result != read_json(reference):
        raise ScoringMismatch('Recomputed metrics differ from scoring.reference_metrics')
    # Verification completes before any new result directory is created.
    output = start_run(config, dataset)
    with (output / 'predictions.jsonl').open('wb') as stream:
        for source in sources:
            with Path(source).open('rb') as original:
                shutil.copyfileobj(original, stream)
            # A missing final newline must not concatenate two JSON records.
            if Path(source).stat().st_size and not Path(source).read_bytes().endswith(b'\n'):
                stream.write(b'\n')
    for row in scored:
        append_json(output / 'scored_predictions.jsonl', row)
    finish_run(config, dataset, output, scored, result)
    write_json(output / 'verification.json', dict(records=len(scored),
        verdicts_identical=config['scoring'].get('verify_existing', True),
        metrics_identical=config['scoring'].get('verify_existing', True), reference_metrics=reference))
    return result

def run(config, dataset):
    # Heavy dependencies enter only the explicitly selected run path.
    from .model import load_model
    from .interface import StateInterface, CachedStateInterface, DecisionInterface, make_adapter
    from .inference import generate, NodeIdGrammar
    from .training import train, load_adapter
    if any(c['adapter'] is not None for c in config['conditions']) and dataset.q is None:
        raise ValueError('Adapter conditions require data.asset_dir with frozen Q/V')
    output = start_run(config, dataset)
    model, tokenizer = load_model(config)
    adjacency = adjacency_text(dataset.adjacency)
    all_scored = []
    geometry = {}
    if 'geometry' in config.get('diagnostics', []):
        from .geometry import geometry_stat
        if dataset.q is None:
            raise ValueError('Geometry diagnostics require an existing map asset')
        geometry['raw_Q'] = geometry_stat(dataset.q, dataset.distances)

    def record_batch(items, generated, condition, checkpoint, path, extra=None):
        scored = []
        for item, generation in zip(items, generated):
            row = dict(item, **generation, condition=condition, checkpoint=checkpoint, **(extra or {}))
            ids = row.get('generated_ids', [])
            end = tokenizer.eos_token_id
            row['generated_token_count'] = ids.index(end) + 1 if end in ids else len(ids)
            append_json(path, row)
            scored.append(score_record(row, dataset.adjacency, dataset.distances, config['scoring']['type']))
        return scored

    # Finish all training and selection before producing any test predictions.
    for condition in config['conditions']:
        if not config['training']['enabled'] or not condition.get('train', condition['adapter'] is not None):
            continue
        changed = stage_config(config, condition, 'train')
        interface_cls = CachedStateInterface if config['training'].get('cache_frozen_prefix') else StateInterface
        interface = interface_cls(changed, model, tokenizer, dataset.q, dataset.rms, adjacency)
        adapter = make_adapter(config, condition['adapter'], model.config.hidden_size, interface.device)
        if condition.get('initial_checkpoint'):
            load_adapter(condition['initial_checkpoint'], adapter, interface.device)
        directory = output / condition['name']
        directory.mkdir()
        items = training_examples(config, dataset, condition)
        validation = training_examples(config, dataset, condition, 'validation')
        def validate(active_interface, active_adapter, values, epoch):
            rows = []
            for batch in chunks(values, config['eval']['batch_size']):
                generation = generate(active_interface, batch, active_adapter)
                for item, answer in zip(batch, generation):
                    raw = dict(item, **answer, condition=condition['name'], checkpoint='validation', epoch=epoch)
                    append_json(directory / 'validation_predictions.jsonl', raw)
                    rows.append(score_record(raw, dataset.adjacency, dataset.distances, 'strict_node'))
            return sum(r['correct'] for r in rows), len(rows)
        train(interface, adapter, items, validation, directory, validate)
        condition['checkpoints'] = {name: str(directory / (name + '.safetensors'))
                                    for name in condition.get('evaluate_checkpoints', ['selected', 'final'])}
        del interface, adapter

    for condition in config['conditions']:
        changed = stage_config(config, condition, 'eval')
        if config['scoring']['type'] == 'choice':
            items_by_task = read_json(config['data']['items_path'])
            items = [item for task in config['eval']['tasks'] for item in items_by_task[task]]
            permutation = read_json(config['data']['permutation_path'])
            interface = DecisionInterface(model, tokenizer, dataset.q, dataset.rms, adjacency, permutation)
        else:
            items = [item for template in config['eval']['templates']
                     for item in examples(dataset.split[config['eval']['split']], config['eval']['tasks'], template)]
            interface = StateInterface(changed, model, tokenizer, dataset.q, dataset.rms, adjacency)
        checkpoints = condition.get('checkpoints', {'frozen' if condition['adapter'] is None else Path(condition['checkpoint']).stem if condition.get('checkpoint') else 'unbound': condition.get('checkpoint')})
        for checkpoint, path in checkpoints.items():
            adapter = None
            if condition['adapter'] is not None:
                adapter = make_adapter(config, condition['adapter'], model.config.hidden_size, model.device)
                if path is None:
                    raise ValueError('Evaluation adapter requires an explicit existing checkpoint')
                load_adapter(path, adapter, model.device).requires_grad_(False)
                if 'geometry' in config.get('diagnostics', []):
                    geometry.setdefault(condition['name'], {})[checkpoint] = geometry_stat(
                        interface.representation(adapter), dataset.distances)
            for batch in chunks(items, config['eval']['batch_size']):
                if config['scoring']['type'] == 'choice':
                    name = condition['decision_condition']
                    # Two-stage reasoning is an existing protocol, explicitly
                    # represented in the task and config, never hidden retries.
                    generation = []
                    for item in batch:
                        analysis = interface.generate([item], name, adapter, analysis=True,
                            reason_tokens=config['generation'].get('reason_tokens', 512))[0] if item['task'] == 'neighbor_reason' else None
                        answer = interface.generate([item], name, adapter, reasons=[analysis['output']] if analysis else None)[0]
                        if analysis:
                            answer['analysis'] = analysis
                        generation.append(answer)
                else:
                    grammar = NodeIdGrammar(tokenizer, len(dataset.adjacency)) if config['eval']['decoding'] == 'node_id_grammar' else None
                    generation = generate(interface, batch, adapter, prefix_allowed_tokens_fn=grammar)
                rows = record_batch(batch, generation, condition['name'], checkpoint,
                    output / 'predictions.jsonl', {'split': config['eval']['split'], 'decoding': config['eval']['decoding']})
                for row in rows:
                    append_json(output / 'scored_predictions.jsonl', row)
                all_scored.extend(rows)
            del adapter
    result = metrics(all_scored, config['scoring']['type'])
    requested = set(config.get('diagnostics', [])) & {'template_consistency', 'swap'}
    if requested:
        result['diagnostics'] = diagnostics(all_scored, dataset.adjacency, dataset.distances, requested)
    if geometry:
        result['geometry'] = geometry
    finish_run(config, dataset, output, all_scored, result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output')
    parser.add_argument('--predictions', nargs='+')
    parser.add_argument('--split')
    parser.add_argument('--asset-dir')
    parser.add_argument('--model-dir')
    parser.add_argument('--model-provenance')
    parser.add_argument('--source-commit', help='Source revision when running a committed git archive')
    parser.add_argument('--blocks-phase', choices=('all', 'train', 'eval', 'summarize'), default='all')
    parser.add_argument('--episode-runs', nargs='+')
    parser.add_argument('--training-run')
    parser.add_argument('--condition', choices=('text', 'text_token'))
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ('output', 'predictions'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key, value in (('split_path', args.split), ('asset_dir', args.asset_dir)):
        if value:
            config['data'][key] = value
    if args.model_dir:
        config['model']['directory'] = args.model_dir
    if args.model_provenance:
        config['model']['provenance'] = args.model_provenance
    if args.source_commit:
        config['source_commit'] = args.source_commit
    if config.get('task_family') == 'blocks':
        config['execution'] = dict(phase=args.blocks_phase, training_run=args.training_run,
                                   condition=args.condition, shard_index=args.shard_index, num_shards=args.num_shards)
    config = resolve_config(config)
    if Path(config['output']).exists():
        raise FileExistsError('Choose a new output directory; existing results are never overwritten')
    if config.get('task_family') == 'blocks':
        if args.blocks_phase == 'summarize':
            from .blocks import collect_blocks_runs
            result = collect_blocks_runs(args.training_run, args.episode_runs, config['output'])
            print(f"blocks complete: {config['output']}; {result}")
            return
        from .blocks_run import run_blocks
        result = run_blocks(config)
        print(f"blocks complete: {config['output']}; {result}")
        return
    dataset = build_dataset(config)
    result = rescore(config, dataset) if config['mode'] == 'rescore' else run(config, dataset)
    print(f"{config['mode']} complete: {result['records']} predictions; {config['output']}")

def legacy_main():
    """Old runner names accept only the new config-driven interface."""
    main()

if __name__ == '__main__':
    main()
