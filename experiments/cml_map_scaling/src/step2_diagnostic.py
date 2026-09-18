"""Matched output-format interventions on completed Step 2 checkpoints."""
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
from .step2_data import adjacency_text, examples, judge, load_assets, summarize, template_consistency
from .step2_model import NodeIdGrammar, StateInterface, make_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='experiments/cml_map_scaling/configs/step2_diagnostic.json')
    parser.add_argument('--eval-batch', type=int, default=8)
    args = parser.parse_args()
    base, output = Path(args.base_run), Path(args.output)
    diagnostic = json.loads(Path(args.config).read_text(encoding='utf-8'))
    config = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    runtime = json.loads((base / 'runtime.json').read_text(encoding='utf-8'))
    split = json.loads((base / 'split.json').read_text(encoding='utf-8'))
    assert runtime['source_commit'] == diagnostic['base_source_commit']
    assert runtime.get('completed_utc'), 'Base training and evaluation must be complete'
    output.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    record = dict(source_commit=source, base_run=str(base), base_source_commit=runtime['source_commit'],
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), eval_batch=args.eval_batch,
                  model_dir=runtime['model_dir'], torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    write_json(output / 'runtime.json', record)
    write_json(output / 'config.json', diagnostic)
    q, adj, distances, rms = load_assets(config, runtime['asset_dir'], ROOT)
    tokenizer = AutoTokenizer.from_pretrained(runtime['model_dir'], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(runtime['model_dir'], dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True).to('cuda')
    items = [item for template in diagnostic['templates']
             for item in examples(split[diagnostic['split']], diagnostic['tasks'], template)]
    summaries = {}
    for variant in diagnostic['variants']:
        changed = copy.deepcopy(config)
        changed['prompts']['system'] += variant['system_suffix']
        interface = StateInterface(changed, model, tokenizer, q, rms, adjacency_text(adj))
        grammar = NodeIdGrammar(tokenizer, len(q)) if variant['node_id_grammar'] else None
        summaries[variant['name']] = {}
        for condition in [dict(name='text_control', adapter=None)] + config['conditions']:
            name = condition['name']
            adapter = None
            if condition['adapter'] is not None:
                adapter = make_adapter(config, condition['adapter'], model.config.hidden_size, 'cuda')
                adapter.load_state_dict(load_file(str(base / name / 'selected.safetensors'), device='cuda'))
            rows = []
            started = time.perf_counter()
            for batch in chunks(items, args.eval_batch):
                generated = interface.generate(batch, adapter, prefix_allowed_tokens_fn=grammar)
                for item, generation in zip(batch, generated):
                    row = dict(judge(item, generation['raw_output'], adj, distances), **generation,
                               condition=name, checkpoint='frozen' if adapter is None else 'selected',
                               variant=variant['name'])
                    rows.append(row)
                    append_json(output / 'predictions.jsonl', row)
            values = {template: summarize([r for r in rows if r['template'] == template])
                      for template in diagnostic['templates']}
            values['template_both_correct'] = template_consistency(rows)
            values['token_limit_counts'] = {template: {
                task: sum(r['hit_token_limit'] for r in rows if r['template'] == template and r['task'] == task)
                for task in diagnostic['tasks']} for template in diagnostic['templates']}
            values['evaluation_seconds'] = time.perf_counter() - started
            summaries[variant['name']][name] = values
            write_json(output / 'summary.json', summaries)
            print(json.dumps(dict(variant=variant['name'], condition=name, summary=values)), flush=True)
            del adapter
    record['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(output / 'runtime.json', record)
    print(json.dumps(dict(status='complete', output=str(output))), flush=True)


if __name__ == '__main__':
    main()
