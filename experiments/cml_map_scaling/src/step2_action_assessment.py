"""Assess saved action continuations using original prompts and actual seen actions."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .run import ROOT, write_json
from .step2 import append_json, chunks
from .step2_data import adjacency_text, examples, fraction, judge, load_assets, summarize
from .step2_model import NodeIdGrammar, StateInterface, make_adapter
from .step2_augmentation import checkpoint_for_condition
from .step2_decision_diagnostics import utc


def assessment_groups(spec, split, training_data):
    """Seen actions are the saved supervised examples, not a new train sample."""
    seen=training_data['actions']
    train_pairs={tuple(sorted(pair)) for pair in split['train']}
    assert all(tuple(sorted((x['u'],x['g']))) in train_pairs for x in seen)
    groups={'seen_actions__canonical':dict(items=seen,decodings=spec['action_decoding'])}
    for template in spec['templates']:
        groups['test_actions__'+template]=dict(items=examples(split['test'],('action',),template),decodings=spec['action_decoding'])
        groups['test_reports__'+template]=dict(items=examples(split['test'],('report_current','report_goal'),template),decodings=spec['report_decoding'])
    return groups


def block_summary(rows):
    result=dict(metrics=summarize(rows),format_valid=fraction(r['format_valid'] for r in rows),
                token_limit=fraction(r['hit_token_limit'] for r in rows))
    labelled=[r for r in rows if 'training_label_match' in r]
    if labelled:
        result['stored_training_label_agreement']=fraction(r['training_label_match'] for r in labelled)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run',type=Path,required=True)
    parser.add_argument('--continuation-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/cml_map_scaling/configs/step2_action_assessment.json')
    args=parser.parse_args()
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    base,continuation,out=args.base_run,args.continuation_run,args.output
    spec=read(args.config);old=read(base/'config.json');runtime=read(base/'runtime.json')
    continuation_runtime=read(continuation/'runtime.json')
    assert runtime['source_commit']==spec['base_source_commit']
    assert old['assets']['node_count'] in spec['node_counts']
    assert continuation_runtime.get('completed_utc')
    assert Path(continuation_runtime['base_run']).resolve()==base.resolve()
    split=read(base/'split.json');groups=assessment_groups(spec,split,read(continuation/'training_data.json'))
    checkpoints={c['name']:checkpoint_for_condition(c,spec,base,continuation) for c in spec['conditions']}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',spec);write_json(out/'items.json',groups)
    binding=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 base_run=str(base),continuation_run=str(continuation),started_utc=utc(),
                 checkpoint_paths={k:str(v) for k,v in checkpoints.items()},gpu=torch.cuda.get_device_name())
    write_json(out/'runtime.json',binding)
    q,adj,distances,rms=load_assets(old,runtime['asset_dir'],ROOT)
    config=copy.deepcopy(old);config['generation']['max_new_tokens']=spec['generation']['max_new_tokens']
    tokenizer=AutoTokenizer.from_pretrained(runtime['model_dir'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(runtime['model_dir'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    interface=StateInterface(config,model,tokenizer,q,rms,adjacency_text(adj))
    summaries={}
    for condition in spec['conditions']:
        name=condition['name'];adapter=make_adapter(old,condition['adapter'],model.config.hidden_size,'cuda')
        adapter.load_state_dict(load_file(str(checkpoints[name]),device='cuda'));adapter.eval().requires_grad_(False)
        summaries[name]={}
        for group,data in groups.items():
            for decoding in data['decodings']:
                rows=[];grammar=NodeIdGrammar(tokenizer,len(q)) if decoding=='node_id_grammar' else None
                for batch in chunks(data['items'],spec['generation']['batch_size']):
                    outputs=interface.generate(batch,adapter,prefix_allowed_tokens_fn=grammar)
                    for item,generation in zip(batch,outputs):
                        row=dict(judge(item,generation['raw_output'],adj,distances),**generation,condition=name,group=group,decoding=decoding)
                        if group.startswith('seen_actions'):
                            row['training_label_match']=row['parsed']==item['target']
                        rows.append(row);append_json(out/'predictions.jsonl',row)
                block=group+'__'+decoding;summaries[name][block]=block_summary(rows)
                write_json(out/'summary.json',summaries)
                print(json.dumps(dict(condition=name,block=block,total=len(rows),summary=summaries[name][block])),flush=True)
        del adapter
    binding['completed_utc']=utc();write_json(out/'runtime.json',binding)
    write_json(out/'completed.json',dict(completed_utc=binding['completed_utc']))


if __name__=='__main__':main()
