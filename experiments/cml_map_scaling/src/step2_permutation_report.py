"""Report physical Q identities in freshly relabeled graphs; no action training."""
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

from grounded_llm.graph_data import examples, relabel_item, relabel_matrix, validate_node_labels
from grounded_llm.graph_prompts import adjacency_text, messages
from grounded_llm.graph_scoring import fraction, judge
from .core import structural_colors
from .run import ROOT, write_json
from .step2 import append_json, chunks, optimizer_for, save_adapter, update
from .step2_data import load_assets, summarize
from .step2_model import CachedStateInterface, NodeIdGrammar, make_adapter


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def permutation_bank(node_count, count, seed, excluded=()):
    rng = np.random.default_rng(seed)
    used = {tuple(labels) for labels in excluded}
    used.add(tuple(range(node_count)))
    result = []
    while len(result) < count:
        labels = tuple(map(int, rng.permutation(node_count)))
        if labels not in used:
            used.add(labels)
            result.append(list(labels))
    return result


def training_plan(items, spec):
    """Match the original full report data and shuffle; only display labels change."""
    training = list(range(len(items))) * spec['report_repeats']
    rng = np.random.default_rng(spec['shuffle_seed'])
    plan = []
    for epoch in range(1, spec['epochs'] + 1):
        ordered = [training[int(i)] for i in rng.permutation(len(training))]
        for step, batch in enumerate(chunks(ordered, spec['report_batch_size']), 1):
            plan.append(dict(epoch=epoch, step=step, indices=batch, permutation_index=len(plan)))
    return plan


class RelabeledStateInterface(CachedStateInterface):
    """Keep items and Q in physical order; change visible adjacency and labels only."""
    def __init__(self, config, model, tokenizer, q, rms, adjacency, labels):
        self.physical_adjacency = adjacency
        self.labels = validate_node_labels(labels, len(q)).copy()
        super().__init__(config, model, tokenizer, q, rms, adjacency_text(relabel_matrix(adjacency, self.labels)))

    def set_labels(self, labels):
        labels = validate_node_labels(labels, len(self.q))
        if np.array_equal(labels, self.labels):
            return
        self.labels = labels.copy()
        self.adjacency = adjacency_text(relabel_matrix(self.physical_adjacency, labels))
        self.prompt_cache.clear()
        self.rebuild_prefix()

    def prompt_ids(self, item, text_control):
        displayed = relabel_item(item, self.labels) if text_control else item
        return super().prompt_ids(displayed, text_control)

    def batch(self, items, adapter, supervised=False, answer_override=None):
        if supervised and answer_override is None:
            if any(item['task'] not in ('report_current', 'report_goal') for item in items):
                raise ValueError('Relabeling contract trains reports only')
            answer_override = [str(self.labels[item['u'] if item['task'] == 'report_current' else item['g']]) for item in items]
        return super().batch(items, adapter, supervised=supervised, answer_override=answer_override)


def score_report(item, generation, labels, adj, distances):
    """Use the existing judge in displayed coordinates, retaining physical IDs."""
    labels = validate_node_labels(labels, len(adj))
    row = judge(relabel_item(item, labels), generation['raw_output'], relabel_matrix(adj, labels), relabel_matrix(distances, labels))
    return dict(row, **generation, physical_u=item['u'], physical_g=item['g'],
                physical_prediction=int(np.argsort(labels)[row['parsed']]) if row['node_id_valid'] else None)


def equivariance(rows):
    key = lambda r: (r['physical_u'], r['physical_g'], r['task'])
    identity = {key(r): r for r in rows if r['group'] == 'identity_test_pairs' and r['decoding'] == 'free'}
    paired = [(r, identity[key(r)]) for r in rows if r['group'] == 'unseen_permutation_test_pairs' and r['decoding'] == 'free']
    return dict(inverse_label_consistency=fraction(a['physical_prediction'] is not None and a['physical_prediction'] == b['physical_prediction'] for a, b in paired),
                both_correct=fraction(a['correct'] and b['correct'] for a, b in paired),
                interpretation='Consistent wrong identities count for consistency, but never for both_correct.')


def prepare(args, spec):
    base, out = args.base_run, args.output
    old = read(base / 'config.json'); binding = read(base / 'runtime.json')
    reference_runtime = read(base / 'train_mlp_report_runtime.json')
    reference_training = read(base / 'mlp_report/training_summary.json')
    assert binding['source_commit'] == spec['base_source_commit'] and old['assets']['node_count'] in spec['node_counts']
    assert reference_runtime.get('completed_utc') and reference_training['selected_epoch'] == spec['training']['original_reference_selected_epoch']
    for field in ('epochs', 'report_batch_size', 'learning_rate', 'tasks'):
        assert old['training'][field] == spec['training'][field], field
    assert old['data']['report_shuffle_seed'] == spec['training']['shuffle_seed']
    assert old['adapter']['initialization_seed'] == spec['initialization_seed']
    split = read(base / 'split.json')
    items = examples(split['train'], spec['training']['tasks'], spec['training']['template'])
    plan = training_plan(items, spec['training'])
    presentations = sum(len(batch['indices']) for batch in plan)
    q, adj, _, _ = load_assets(old, binding['asset_dir'], ROOT)
    colors = structural_colors(adj)
    if len(set(colors)) != len(adj):
        raise ValueError('Color refinement did not distinguish all physical nodes; audit automorphisms before label training.')
    train = permutation_bank(len(q), len(plan), spec['relabeling']['train_seed'])
    test = permutation_bank(len(q), spec['relabeling']['test_permutations'], spec['relabeling']['test_seed'], train)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', spec); write_json(out / 'split.json', split)
    write_json(out / 'training_items.json', items); write_json(out / 'training_plan.json', plan)
    write_json(out / 'permutations.json', dict(identity=list(range(len(q))), train=train, test=test))
    write_json(out / 'graph.json', read(base / 'graph.json'))
    write_json(out / 'runtime.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               base_run=str(base), node_count=len(q), prepared_utc=utc(), sample_presentations=presentations,
               optimizer_updates=len(plan), structural_color_classes=len(set(colors)), structural_signatures_visible=False,
               train_permutations=len(train), test_permutations=len(test), training_microbatch=reference_runtime['microbatch'],
               initialization=spec['initialization'], initialization_seed=spec['initialization_seed'],
               reference_checkpoint=str(base / 'mlp_report/selected.safetensors'),
               reference_identity_predictions=str(base / 'eval_mlp_report/reports.jsonl')))
    print(json.dumps(dict(status='prepared', sample_presentations=presentations, optimizer_updates=len(plan))), flush=True)


def worker(args):
    root = args.output; spec = read(root / 'config.json'); binding = read(root / 'runtime.json')
    base = Path(binding['base_run']); old = read(base / 'config.json'); runtime = read(base / 'runtime.json')
    items = read(root / 'training_items.json'); plan = read(root / 'training_plan.json'); bank = read(root / 'permutations.json')
    directory = root / 'training'; directory.mkdir(exist_ok=False)
    worker_runtime = dict(source_commit=binding['source_commit'], started_utc=utc(), condition=spec['training_condition'], gpu=torch.cuda.get_device_name())
    write_json(directory / 'runtime.json', worker_runtime)
    q, adj, distances, rms = load_assets(old, runtime['asset_dir'], ROOT)
    config = copy.deepcopy(old)
    config['training'].update({field: spec['training'][field] for field in ('epochs', 'report_batch_size', 'learning_rate', 'tasks')})
    config['adapter']['initialization_seed'] = spec['initialization_seed']
    config['generation']['max_new_tokens'] = spec['evaluation']['max_new_tokens']
    tokenizer = AutoTokenizer.from_pretrained(runtime['model_dir'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(runtime['model_dir'], dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda')
    interface = RelabeledStateInterface(config, model, tokenizer, q, rms, adj, bank['identity'])
    adapter = make_adapter(config, spec['adapter'], model.config.hidden_size, 'cuda')
    save_adapter(directory / 'initial.safetensors', adapter)
    worker_runtime.update(initialization=spec['initialization'], initialization_seed=spec['initialization_seed'])
    optimizer = optimizer_for(config, adapter); total = count = 0; started = time.monotonic()
    write_json(directory / 'runtime.json', worker_runtime)
    for batch in plan:
        interface.set_labels(bank['train'][batch['permutation_index']])
        selected = [items[index] for index in batch['indices']]
        loss, grad = update(interface, adapter, optimizer, selected, binding['training_microbatch'])
        append_json(directory / 'train.jsonl', dict(epoch=batch['epoch'], step=batch['step'], samples=len(selected), loss=loss,
                    gradient_norm=grad, permutation_index=batch['permutation_index']))
        if batch['step'] == 1 or batch['step'] % 10 == 0:
            print(json.dumps(dict(epoch=batch['epoch'], step=batch['step'], loss=loss)), flush=True)
        total += loss * len(selected); count += len(selected)
        if batch is plan[-1] or plan[batch['permutation_index'] + 1]['epoch'] != batch['epoch']:
            print(json.dumps(dict(epoch=batch['epoch'], loss=total / count, seconds=time.monotonic() - started)), flush=True)
            total = count = 0
    save_adapter(directory / 'final.safetensors', adapter)
    write_json(directory / 'training_summary.json', dict(sample_presentations=binding['sample_presentations'], optimizer_updates=len(plan), train_seconds=time.monotonic()-started))
    del optimizer
    worker_runtime['training_completed_utc'] = utc(); write_json(directory / 'runtime.json', worker_runtime)
    adapter.eval().requires_grad_(False)
    test_items = examples(read(root / 'split.json')['test'], spec['training']['tasks'], spec['evaluation']['template'])
    seen = items[:round(len(q) * spec['evaluation']['seen_examples_per_node'])]
    seen_index = spec['relabeling']['seen_permutation_index']
    actually_seen = [items[i] for i in plan[seen_index]['indices']]
    tests = [('identity_seen_pairs', 'identity', bank['identity'], seen, ['free']),
             ('identity_test_pairs', 'identity', bank['identity'], test_items, ['free', 'node_id_grammar']),
             ('seen_permutation_seen_pairs', 'train_' + str(seen_index), bank['train'][seen_index], actually_seen, ['free'])]
    for index, labels in enumerate(bank['test']):
        tests.extend([('unseen_permutation_seen_pairs', 'test_' + str(index), labels, seen, ['free']),
                      ('unseen_permutation_test_pairs', 'test_' + str(index), labels, test_items, ['free', 'node_id_grammar'])])
    reference = make_adapter(config, spec['adapter'], model.config.hidden_size, 'cuda')
    reference.load_state_dict(load_file(binding['reference_checkpoint'], device='cuda')); reference.eval().requires_grad_(False)
    summaries = {}
    for checkpoint, active in [('new_permuted', adapter), ('existing_fixed_reference', reference)]:
        output = root / 'evaluation' / checkpoint; output.mkdir(parents=True, exist_ok=False)
        all_rows = []; blocks = {}
        if checkpoint == 'existing_fixed_reference':
            source = Path(binding['reference_identity_predictions'])
            original = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
            assert {(r['u'], r['g'], r['task']) for r in original} == {(r['u'], r['g'], r['task']) for r in test_items}
            for generation in original:
                item = {field:generation[field] for field in ('u', 'g', 'task', 'template')}
                fields = {field:generation[field] for field in ('raw_output', 'generated_ids', 'native_end_seen', 'hit_token_limit')}
                row = dict(score_report(item, fields, bank['identity'], adj, distances), group='identity_test_pairs', permutation='identity', decoding='free', reused_from=str(source))
                all_rows.append(row); append_json(output / 'predictions.jsonl', row)
            blocks['identity_test_pairs__identity__free'] = dict(metrics=summarize(all_rows), reused_from=str(source))
        active_tests = tests if checkpoint == 'new_permuted' else [t for t in tests if t[0] == 'unseen_permutation_test_pairs']
        for group, name, labels, selected, decodings in active_tests:
            interface.set_labels(labels)
            write_json(output / (group + '__' + name + '_prompt.json'), messages(config, interface.adjacency, selected[0]))
            for decoding in decodings:
                rows = []; grammar = NodeIdGrammar(tokenizer, len(q)) if decoding == 'node_id_grammar' else None
                for batch in chunks(selected, spec['evaluation']['batch_size']):
                    for item, generation in zip(batch, interface.generate(batch, active, prefix_allowed_tokens_fn=grammar)):
                        row = dict(score_report(item, generation, labels, adj, distances), group=group, permutation=name, decoding=decoding)
                        rows.append(row); all_rows.append(row); append_json(output / 'predictions.jsonl', row)
                block = group + '__' + name + '__' + decoding
                blocks[block] = dict(metrics=summarize(rows), token_limit=fraction(row['hit_token_limit'] for row in rows))
                write_json(output / 'summary.json', dict(blocks=blocks))
                print(json.dumps(dict(checkpoint=checkpoint, block=block, summary=blocks[block])), flush=True)
        summaries[checkpoint] = dict(blocks=blocks, equivariance=equivariance(all_rows), total_records=len(all_rows), reused_records=sum('reused_from' in r for r in all_rows))
        write_json(output / 'summary.json', summaries[checkpoint])
        write_json(output / 'completed.json', dict(completed_utc=utc()))
    write_json(root / 'summary.json', dict(checkpoints=summaries, training=read(directory / 'training_summary.json')))
    worker_runtime['completed_utc'] = utc(); write_json(directory / 'runtime.json', worker_runtime)
    write_json(root / 'completed.json', dict(completed_utc=worker_runtime['completed_utc']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/cml_map_scaling/configs/step2_permutation_report.json')
    parser.add_argument('--base-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        if args.base_run is None:
            parser.error('--prepare requires --base-run')
        prepare(args, read(args.config))
    else:
        worker(args)


if __name__ == '__main__':
    main()
