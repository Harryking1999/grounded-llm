"""Evaluate complete text inputs with and without existing roadmap adapters."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .run import ROOT, write_json
from .step2 import append_json, chunks
from .step2_data import adjacency_text, examples, load_assets
from grounded_llm.graph_scoring import extract_decision, score_decision, summarize_actions
from .step2_model import StateInterface, make_adapter


def augmentation_messages(spec, adjacency, item, with_roadmap, node_count=32):
    prompts = spec['prompts']
    block = prompts['roadmap_block'].format(current_slot='<|cml_current_state|>',
                                             goal_slot='<|cml_goal_state|>') if with_roadmap else ''
    user = prompts['user'].format(max_node_id=node_count - 1, adjacency=adjacency, u=item['u'], g=item['g'], roadmap_block=block)
    return [{'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': user}]


class AugmentedInterface(StateInterface):
    def __init__(self, config, spec, *args):
        self.spec = spec
        super().__init__(config, *args)

    def prompt_ids(self, item, text_control):
        # Every prompt includes explicit IDs, even when an adapter is present.
        key = (item['u'], item['g'], item['task'], item['template'], text_control)
        if key not in self.prompt_cache:
            ids = self.tokenizer.apply_chat_template(
                augmentation_messages(self.spec, self.adjacency, item, not text_control,
                                      self.config['assets']['node_count']),
                tokenize=True, add_generation_prompt=True, return_dict=False)
            if not text_control:
                assert all(ids.count(slot) == 1 for slot in self.slot_ids)
            self.prompt_cache[key] = ids
        return self.prompt_cache[key]


def checkpoint_for_condition(condition, spec, base, continuation=None):
    """Resolve one adapter checkpoint from the formal condition, never from a run name guess."""
    if condition['adapter'] is None:
        return None
    source = condition.get('checkpoint_source', 'base')
    if source == 'base':
        root = base
    elif source == 'continuation' and continuation is not None:
        root = continuation
    else:
        raise ValueError(f'Unavailable checkpoint source: {source}')
    name = condition.get('checkpoint_condition', condition.get('base_condition'))
    if not name:
        raise ValueError('Adapter condition needs a checkpoint condition')
    checkpoint = condition.get('checkpoint', spec['checkpoint'])
    path = root / name / (checkpoint + '.safetensors')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def pending_items(items, rows):
    """Resume one condition from complete saved records, keeping original order."""
    done = {(r['u'],r['g']) for r in rows}
    return [r for r in items if (r['u'],r['g']) not in done]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/cml_map_scaling/configs/step2_augmentation.json')
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--eval-batch', type=int, default=16)
    parser.add_argument('--continuation-run', help='Existing matched action pilot run, when requested by the config')
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted shard without regenerating recorded items')
    args = parser.parse_args()
    base, output = Path(args.base_run), Path(args.output)
    continuation = Path(args.continuation_run) if args.continuation_run else None
    spec = json.loads(Path(args.config).read_text(encoding='utf-8'))
    old = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    base_runtime = json.loads((base / 'runtime.json').read_text(encoding='utf-8'))
    split = json.loads((base / 'split.json').read_text(encoding='utf-8'))
    assert base_runtime['source_commit'] == spec['base_source_commit']
    assert 0 <= args.shard_index < args.num_shards
    if continuation is not None:
        continuation_runtime = json.loads((continuation/'runtime.json').read_text(encoding='utf-8'))
        continuation_config = json.loads((continuation/'config.json').read_text(encoding='utf-8'))
        assert continuation_runtime.get('completed_utc')
        assert Path(continuation_runtime['base_run']).resolve() == base.resolve()
        assert continuation_config['base_source_commit'] == base_runtime['source_commit']
    checkpoint_paths = {c['name']:checkpoint_for_condition(c,spec,base,continuation) for c in spec['conditions']}
    if args.resume:
        assert json.loads((output/'config.json').read_text(encoding='utf-8')) == spec
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', spec)
    runtime = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   base_source_commit=base_runtime['source_commit'], base_run=str(base),
                   shard_index=args.shard_index, num_shards=args.num_shards, eval_batch=args.eval_batch,
                   torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                   checkpoint_paths={k:str(v) if v else None for k,v in checkpoint_paths.items()},
                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if args.resume:
        previous = json.loads((output/'runtime.json').read_text(encoding='utf-8'))
        assert previous['base_run'] == str(base) and previous['checkpoint_paths'] == runtime['checkpoint_paths']
        assert (previous['shard_index'],previous['num_shards']) == (args.shard_index,args.num_shards)
        previous.setdefault('resumptions', []).append(dict(source_commit=runtime['source_commit'],
            started_utc=runtime['started_utc'],eval_batch=args.eval_batch,gpu=runtime['gpu']))
        runtime = previous
    write_json(output / 'runtime.json', runtime)
    q, adj, distances, rms = load_assets(old, base_runtime['asset_dir'], ROOT)
    config = copy.deepcopy(old)
    config['generation'].update(spec['generation'])
    tokenizer = AutoTokenizer.from_pretrained(base_runtime['model_dir'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base_runtime['model_dir'], dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True).to('cuda')
    interface = AugmentedInterface(config, spec, model, tokenizer, q, rms, adjacency_text(adj))
    all_items = examples(split[spec['split']], spec['tasks'])
    items = all_items[args.shard_index::args.num_shards]
    write_json(output / 'items.json', items)
    write_json(output / 'prompt_examples.json', {name: augmentation_messages(spec, interface.adjacency, items[0], enabled)
                                                for name,enabled in (('baseline',False),('augmentation',True))})
    existing = [json.loads(line) for line in (output/'predictions.jsonl').read_text(encoding='utf-8').splitlines()] if args.resume and (output/'predictions.jsonl').exists() else []
    keys = [(r['condition'],r['u'],r['g']) for r in existing]
    assert len(keys) == len(set(keys))
    expected = {(c['name'],r['u'],r['g']) for c in spec['conditions'] for r in items}
    assert set(keys).issubset(expected)
    summaries = {}
    for condition in spec['conditions']:
        name = condition['name']
        rows = [r for r in existing if r['condition'] == name]
        remaining = pending_items(items, rows)
        if not remaining:
            summaries[name] = summarize_actions(rows)
            continue
        adapter = None
        if condition['adapter'] is not None:
            adapter = make_adapter(old, condition['adapter'], model.config.hidden_size, 'cuda')
            checkpoint = checkpoint_paths[name]
            adapter.load_state_dict(load_file(str(checkpoint), device='cuda'))
            adapter.requires_grad_(False)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for batch_index, batch in enumerate(chunks(remaining, args.eval_batch)):
            for item, generation in zip(batch, interface.generate(batch, adapter)):
                ids = generation['generated_ids']
                count = ids.index(interface.end_id)+1 if interface.end_id in ids else len(ids)
                row = dict(score_decision(item, generation['raw_output'], adj, distances), **generation,
                           condition=name, checkpoint='frozen' if adapter is None else checkpoint.stem,
                           shard_index=args.shard_index, generated_token_count=count)
                rows.append(row)
                append_json(output / 'predictions.jsonl', row)
            print(json.dumps(dict(condition=name, shard=args.shard_index, completed=len(rows), total=len(items),
                                  batch=batch_index)), flush=True)
        summaries[name] = dict(summarize_actions(rows), evaluation_seconds=time.perf_counter()-started,
                               peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(output / 'summary.json', summaries)
        print(json.dumps(dict(condition=name, summary=summaries[name])), flush=True)
        del adapter
    runtime['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(output / 'runtime.json', runtime)
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)


if __name__ == '__main__':
    main()
