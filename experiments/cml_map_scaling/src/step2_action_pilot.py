"""Small action-supervised continuation and matched report-only control."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .run import ROOT, write_json
from .step2 import append_json, chunks, optimizer_for, save_adapter, update
from .step2_data import adjacency_text, examples, geometry_stat, judge, load_assets, summarize
from .step2_model import CachedStateInterface, NodeIdGrammar, StateInterface, make_adapter


def make_training_data(spec, split, adj, distances):
    action_spec = spec['action_data']
    pool = split[action_spec['source_split']]
    order = np.random.default_rng(action_spec['pair_seed']).permutation(len(pool))
    pair_count = action_spec['unordered_pairs'] if 'unordered_pairs' in action_spec else round(len(adj) * action_spec['unordered_pairs_per_node'])
    pairs = [pool[int(i)] for i in order[:pair_count]]
    actions = examples(pairs, ('action',))
    rng = np.random.default_rng(action_spec['label_seed'])
    for item in actions:
        valid = np.flatnonzero(adj[item['u']] & (distances[:, item['g']] < distances[item['u'], item['g']]))
        item['target'] = int(rng.choice(valid))
    report_spec = spec['report_replay']
    pool = examples(split[report_spec['source_split']])
    order = np.random.default_rng(report_spec['seed']).permutation(len(pool))
    selected, keys = [], set()
    for index in order:
        item = pool[int(index)]
        target = item['u'] if item['task'] == 'report_current' else item['g']
        key = (item['task'], target)
        if key not in keys:
            selected.append(int(index))
            keys.add(key)
    selected_set = set(selected)
    report_count = report_spec['examples'] if 'examples' in report_spec else round(len(adj) * report_spec['examples_per_node'])
    selected += [int(i) for i in order if int(i) not in selected_set][:report_count - len(selected)]
    reports = [pool[i] for i in selected]
    assert len(reports) == len(actions) == 2 * pair_count and len(keys) == len(adj) * 2
    return dict(action_pairs=pairs, actions=actions, reports=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='experiments/cml_map_scaling/configs/step2_action_pilot.json')
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--eval-batch', type=int, default=8)
    parser.add_argument('--train-only', action='store_true', help='Leave action assessment to matched full-text evaluator')
    args = parser.parse_args()
    base, output = Path(args.base_run), Path(args.output)
    spec = json.loads(Path(args.config).read_text(encoding='utf-8'))
    original = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    base_runtime = json.loads((base / 'runtime.json').read_text(encoding='utf-8'))
    split = json.loads((base / 'split.json').read_text(encoding='utf-8'))
    assert base_runtime['source_commit'] == spec['base_source_commit']
    # Scaling case runtimes record preparation, while checkpoint completion is
    # evidenced by the selected weights and their training summary.
    assert (base / spec['base_condition'] / (spec['base_checkpoint'] + '.safetensors')).is_file()
    assert (base / spec['base_condition'] / 'training_summary.json').is_file()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', spec)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    runtime = dict(source_commit=source, base_run=str(base), base_source_commit=base_runtime['source_commit'],
                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   microbatch=args.microbatch, eval_batch=args.eval_batch,
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    write_json(output / 'runtime.json', runtime)
    q, adj, distances, rms = load_assets(original, base_runtime['asset_dir'], ROOT)
    data = make_training_data(spec, split, adj, distances)
    write_json(output / 'training_data.json', data)
    config = copy.deepcopy(original)
    config['training'].update({k: spec['training'][k] for k in ('epochs', 'report_batch_size', 'learning_rate')})
    config['training']['tasks'] = ['report_current', 'report_goal', 'action']
    tokenizer = AutoTokenizer.from_pretrained(base_runtime['model_dir'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base_runtime['model_dir'], dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True).to('cuda')
    interface_class = CachedStateInterface if spec['training'].get('cache_frozen_prefix') else StateInterface
    interface = interface_class(config, model, tokenizer, q, rms, adjacency_text(adj))
    base_weights = load_file(str(base / spec['base_condition'] / (spec['base_checkpoint'] + '.safetensors')), device='cuda')
    training_summaries = {}
    for condition in spec['conditions']:
        name = condition['name']
        directory = output / name
        directory.mkdir()
        adapter = make_adapter(config, 'mlp', model.config.hidden_size, 'cuda')
        adapter.load_state_dict(base_weights)
        optimizer = optimizer_for(config, adapter)
        second = data['actions'] if condition['pool'] in ('report_replay_plus_80_actions', 'report_replay_plus_actions') else data['reports']
        training = data['reports'] + second
        rng = np.random.default_rng(spec['training']['shuffle_seed'])
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, spec['training']['epochs'] + 1):
            ordered = [training[int(i)] for i in rng.permutation(len(training))]
            total = 0.0
            for step, batch in enumerate(chunks(ordered, spec['training']['report_batch_size']), 1):
                loss, grad = update(interface, adapter, optimizer, batch, args.microbatch)
                total += loss * len(batch)
                append_json(directory / 'train.jsonl', dict(epoch=epoch, step=step, samples=len(batch), loss=loss, gradient_norm=grad))
            print(json.dumps(dict(condition=name, epoch=epoch, loss=total / len(training))), flush=True)
        save_adapter(directory / 'final.safetensors', adapter)
        training_summaries[name] = dict(train_seconds=time.perf_counter() - started,
                                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                                        sample_presentations=len(training) * spec['training']['epochs'])
        write_json(directory / 'training_summary.json', training_summaries[name])
        del adapter, optimizer
    if args.train_only:
        write_json(output / 'training_summary.json', training_summaries)
        runtime['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(output / 'runtime.json', runtime)
        print(json.dumps(dict(status='training_complete', output=str(output))), flush=True)
        return
    # Both fixed-budget continuations finish before either test result is produced.
    summaries = {}
    for condition in spec['conditions']:
        name = condition['name']
        directory = output / name
        adapter = make_adapter(config, 'mlp', model.config.hidden_size, 'cuda')
        adapter.load_state_dict(load_file(str(directory / 'final.safetensors'), device='cuda'))
        values = dict(training=training_summaries[name], geometry=geometry_stat(interface.representation(adapter), distances))
        for decoding in spec['evaluation']['decoding']:
            grammar = NodeIdGrammar(tokenizer, len(q)) if decoding == 'node_id_grammar' else None
            values[decoding] = {}
            for split_name in ('validation', 'test'):
                templates = spec['evaluation'][split_name + '_templates']
                items = [item for template in templates for item in examples(split[split_name], spec['evaluation']['tasks'], template)]
                rows = []
                for batch in chunks(items, args.eval_batch):
                    generated = interface.generate(batch, adapter, prefix_allowed_tokens_fn=grammar)
                    for item, generation in zip(batch, generated):
                        row = dict(judge(item, generation['raw_output'], adj, distances), **generation,
                                   condition=name, checkpoint='final', decoding=decoding, split=split_name)
                        rows.append(row)
                        append_json(directory / 'predictions.jsonl', row)
                values[decoding][split_name] = {t: summarize([r for r in rows if r['template'] == t]) for t in templates}
            print(json.dumps(dict(condition=name, decoding=decoding, test=values[decoding]['test'])), flush=True)
        write_json(directory / 'summary.json', values)
        summaries[name] = values
        del adapter
    write_json(output / 'summary.json', summaries)
    runtime['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(output / 'runtime.json', runtime)
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)


if __name__ == '__main__':
    main()
