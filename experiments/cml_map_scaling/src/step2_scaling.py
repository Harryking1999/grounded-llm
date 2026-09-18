"""Train graph-specific report adapters, then evaluate matched roadmap augmentation."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .run import ROOT, write_json
from .step2 import append_json, chunks, predict, train_condition
from .step2_augmentation import AugmentedInterface, augmentation_messages, score_decision, summarize_actions
from .step2_data import adjacency_text, examples, geometry_stat, load_assets, make_split, summarize
from .step2_model import CachedReportInterface, StateInterface, make_adapter


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def case_config(spec, case, graph_reference):
    config = copy.deepcopy(read(ROOT/spec['base_training_config']))
    config['study'] = spec['study']
    config['assets'].update(case=case['case'],node_count=case['node_count'],graph_reference=str(graph_reference))
    config['data'].update(sample_subset=True,split_seed=spec['data']['split_seed'],
        unordered_pair_counts=dict(train=case['train_pairs'],validation=spec['data']['validation_pairs'],test=spec['data']['test_pairs']))
    config['outputs']['root'] = 'runs/cml_step2_scaling'
    return config


def prepare(args,spec):
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'scaling_config.json',spec)
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for case in spec['assets']['cases']:
        folder = args.output/case['case']
        folder.mkdir()
        assets = args.asset_root/spec['assets']['condition']/case['case']
        with np.load(assets/'inputs.npz',allow_pickle=False) as data:
            adjacency = data['adjacency'].astype(bool)
        graph = dict(neighbors={str(i):np.flatnonzero(row).tolist() for i,row in enumerate(adjacency)})
        write_json(folder/'graph.json',graph)
        config = case_config(spec,case,folder/'graph.json')
        q,adj,dist,rms = load_assets(config,assets,ROOT)
        split = make_split(config)
        write_json(folder/'config.json',config)
        write_json(folder/'split.json',split)
        (folder/'adjacency.txt').write_text(adjacency_text(adj)+'\n',encoding='utf-8')
        write_json(folder/'runtime.json',dict(source_commit=source,asset_source_revision=spec['assets']['step1_source_commit'],
            asset_dir=str(assets),model_dir=str(args.model_dir),q_global_rms=rms,
            node_count=len(q),graph_distance_max=int(dist.max()),prepared_utc=utc(),
            train_report_examples=4*len(split['train']),test_action_examples=2*len(split['test']),
            unused_unordered_pairs=len(q)*(len(q)-1)//2-sum(map(len,split.values()))))
        print(json.dumps(dict(status='prepared',case=case['case'],reports=4*len(split['train']))),flush=True)


def probe(interface,adapter,folder):
    """One real-model equivalence/timing check before using cached training."""
    n = len(interface.q)
    items = [dict(u=0,g=n-1,task='report_current',template='canonical'),
             dict(u=n-1,g=1,task='report_goal',template='canonical')]
    observations = []
    gradients = []
    for cached in (False,True,True):
        adapter.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        losses = interface.losses(items,adapter) if cached else StateInterface.losses(interface,items,adapter)
        losses.mean().backward()
        torch.cuda.synchronize()
        observations.append(dict(cached=cached,losses=losses.detach().float().cpu().tolist(),
                                 seconds=time.perf_counter()-started))
        gradients.append(torch.cat([p.grad.detach().float().flatten() for p in adapter.parameters()]))
    cosine = [float(torch.nn.functional.cosine_similarity(gradients[0],g,dim=0)) for g in gradients[1:]]
    result = dict(dtype=str(interface.dtype),observations=observations,gradient_cosines=cosine,prefix_tokens=interface.prefix_length,
                  prefix_length_after=[int(k.shape[-2]) for k,v in interface.prefix_kv],
                  frozen_gradients_absent=all(p.grad is None for p in interface.model.parameters()))
    write_json(folder/'prefix_probe.json',result)
    np.testing.assert_allclose(observations[0]['losses'],observations[1]['losses'],rtol=1e-4,atol=1e-4)
    np.testing.assert_allclose(observations[1]['losses'],observations[2]['losses'],rtol=1e-5,atol=1e-5)
    assert min(cosine)>.9999 and result['frozen_gradients_absent']
    assert all(length==interface.prefix_length for length in result['prefix_length_after'])
    print(json.dumps(result),flush=True)


def evaluate(interface,config,spec,condition,split,q,adj,dist,folder,args):
    # No action outputs until both checkpoint selections for this graph are fixed.
    for name in spec['training']['conditions']:
        if not (folder/name/'training_summary.json').exists():
            raise RuntimeError(f'Finish training {name} before action evaluation')
    output = folder/('eval_'+condition['name'])
    output.mkdir(exist_ok=False)
    adapter = make_adapter(config,condition['adapter'],interface.model.config.hidden_size,'cuda')
    adapter.load_state_dict(load_file(str(folder/condition['name']/'selected.safetensors'),device='cuda'))
    adapter.requires_grad_(False)
    reports = predict(interface,adapter,examples(split['test']),args.eval_batch,adj,dist,
                      condition['name'],'selected',output/'reports.jsonl')
    report_summary = summarize(reports)
    augmentation = copy.deepcopy(read(ROOT/spec['augmentation_config']))
    augmentation.update(base_source_commit=read(folder/'runtime.json')['source_commit'],base_run=str(folder))
    write_json(output/'augmentation_config.json',augmentation)
    aug = AugmentedInterface(config,augmentation,interface.model,interface.tokenizer,q,
                             read(folder/'runtime.json')['q_global_rms'],adjacency_text(adj))
    aug.generation.max_new_tokens = augmentation['generation']['max_new_tokens']
    items = examples(split['test'],('action',))
    conditions = [('text_baseline',None)] if condition['adapter']=='linear' else []
    conditions.append(('text_plus_'+condition['adapter'],adapter))
    summaries = {}
    write_json(output/'prompt_examples.json',{name:augmentation_messages(augmentation,aug.adjacency,items[0],a is not None,len(q)) for name,a in conditions})
    for name,active_adapter in conditions:
        rows=[]
        started=time.perf_counter()
        for batch in chunks(items,args.eval_batch):
            for item,generation in zip(batch,aug.generate(batch,active_adapter)):
                ids=generation['generated_ids']
                count=ids.index(aug.end_id)+1 if aug.end_id in ids else len(ids)
                row=dict(score_decision(item,generation['raw_output'],adj,dist),**generation,
                         condition=name,generated_token_count=count)
                rows.append(row)
                append_json(output/'predictions.jsonl',row)
            print(json.dumps(dict(case=folder.name,condition=name,completed=len(rows),total=len(items))),flush=True)
        summaries[name]=dict(summarize_actions(rows),seconds=time.perf_counter()-started)
        write_json(output/'summary.json',dict(actions=summaries,reports=report_summary,
            geometry=dict(raw_Q=geometry_stat(q,dist),selected=geometry_stat(interface.representation(adapter),dist)),
            semantic_review_required=True))
    write_json(output/'completed.json',dict(completed_utc=utc()))


def run_queue(args,spec):
    evaluation_only=args.mode=='evaluate-queue'
    logs=args.output/('migration_eval_logs' if evaluation_only else 'logs')
    logs.mkdir(exist_ok=False)
    status=dict(phase='evaluating' if evaluation_only else 'training',started_utc=utc(),gpus=args.gpus,
                microbatch=args.microbatch,eval_batch=args.eval_batch)
    write_json(args.output/'queue_status.json',status)

    def launch(gpu,mode,n,condition):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
        command=[sys.executable,'-u','-m','experiments.cml_map_scaling.src.step2_scaling',
                 '--config',str(args.config),'--output',str(args.output),'--mode',mode,
                 '--node-count',str(n),'--condition',condition,'--microbatch',str(args.microbatch),
                 '--eval-batch',str(args.eval_batch)]
        with (logs/f'{mode}_{n}_{condition}.log').open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            print(json.dumps(dict(phase=mode,nodes=n,condition=condition,gpu=gpu,pid=process.pid)),flush=True)
            result=process.wait()
        if result:
            raise RuntimeError(f'{mode} {n} {condition} exited {result}; see its log')

    def train_worker(gpu,condition):
        for case in spec['assets']['cases']:
            launch(gpu,'train',case['node_count'],condition)

    pending=Queue()
    for case in reversed(spec['assets']['cases']):
        for condition in spec['training']['conditions']:
            pending.put((case['node_count'],condition))

    def eval_worker(gpu):
        while True:
            try:
                n,condition=pending.get_nowait()
            except Empty:
                return
            launch(gpu,'evaluate',n,condition)

    try:
        if not evaluation_only:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(train_worker,gpu,condition) for gpu,condition in zip(args.gpus,spec['training']['conditions'])]
                for future in futures:
                    future.result()
        status.update(phase='evaluating')
        if not evaluation_only:
            status['training_completed_utc']=utc()
        write_json(args.output/'queue_status.json',status)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(eval_worker,gpu) for gpu in args.gpus]
            for future in futures:
                future.result()
        status.update(phase='complete_pending_semantic_review',completed_utc=utc())
    except Exception as error:
        status.update(phase='failed',error=str(error),failed_utc=utc())
        write_json(args.output/'queue_status.json',status)
        raise
    write_json(args.output/'queue_status.json',status)
    print(json.dumps(status),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/cml_map_scaling/configs/step2_scaling.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('prepare','probe','train','evaluate','queue','evaluate-queue'),required=True)
    parser.add_argument('--node-count',type=int,choices=(64,128,256))
    parser.add_argument('--condition',choices=('linear_report','mlp_report'),default='linear_report')
    parser.add_argument('--asset-root',type=Path)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--microbatch',type=int,default=16)
    parser.add_argument('--eval-batch',type=int,default=8)
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1])
    args=parser.parse_args()
    args.output=args.output.resolve()
    spec=read(args.config)
    if args.mode=='prepare':
        if args.asset_root is None or args.model_dir is None:
            parser.error('prepare needs --asset-root and --model-dir')
        prepare(args,spec)
        return
    if args.mode in ('queue','evaluate-queue'):
        run_queue(args,spec)
        return
    if args.node_count is None:
        parser.error('--node-count is required')
    case=next(c for c in spec['assets']['cases'] if c['node_count']==args.node_count)
    folder=args.output/case['case']
    config,runtime,split=read(folder/'config.json'),read(folder/'runtime.json'),read(folder/'split.json')
    q,adj,dist,rms=load_assets(config,runtime['asset_dir'],ROOT)
    tokenizer=AutoTokenizer.from_pretrained(runtime['model_dir'],local_files_only=True)
    # FP32 distinguishes cache-logic errors from BF16 GEMM/attention rounding.
    # Use the 64-node graph for this probe; FP32 full-sequence backward at 256
    # nodes exceeds the available GPU memory. Formal training remains BF16.
    model=AutoModelForCausalLM.from_pretrained(runtime['model_dir'],dtype=torch.float32 if args.mode=='probe' else torch.bfloat16,
                                              attn_implementation='sdpa',local_files_only=True).to('cuda')
    interface_class=StateInterface if args.mode=='evaluate' else CachedReportInterface
    interface=interface_class(config,model,tokenizer,q,rms,adjacency_text(adj))
    condition=next(c for c in config['conditions'] if c['name']==args.condition)
    phase=dict(started_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               mode=args.mode,condition=args.condition,microbatch=args.microbatch,eval_batch=args.eval_batch,
               gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
    if args.mode=='probe':
        probe(interface,make_adapter(config,condition['adapter'],model.config.hidden_size,'cuda'),folder)
        return
    phase_path=folder/(args.mode+'_'+args.condition+'_runtime.json')
    write_json(phase_path,phase)
    if args.mode=='train':
        phase['result']=train_condition(interface,condition,split,adj,dist,folder,args)
    else:
        evaluate(interface,config,spec,condition,split,q,adj,dist,folder,args)
    phase['completed_utc']=utc()
    write_json(phase_path,phase)
    print(json.dumps(dict(status='complete',case=case['case'],mode=args.mode,condition=args.condition)),flush=True)


if __name__=='__main__':
    main()
