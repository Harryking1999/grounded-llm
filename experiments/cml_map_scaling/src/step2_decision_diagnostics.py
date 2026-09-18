"""Fixed-checkpoint decision completion and distance-relation diagnostics."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .core import shortest_distances
from .run import ROOT, write_json
from .step2 import append_json, chunks
from .step2_data import adjacency_text, examples
from .step2_model import make_adapter
from .step2_augmentation import checkpoint_for_condition


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def graph_data(base):
    graph=read(base/'graph.json')['neighbors']
    adj=np.zeros((len(graph),len(graph)),dtype=bool)
    for a,bs in graph.items():
        adj[int(a),bs]=True
    return adj,shortest_distances(adj)


def make_items(base,spec):
    adj,dist=graph_data(base)
    anchors=examples(read(base/'split.json')['test'],('action',))
    rng=np.random.default_rng(spec['seed'])
    result={t:[] for t in spec['action_tasks']+spec['probe_tasks']}
    for anchor_index,x in enumerate(anchors):
        a,g=x['u'],x['g']
        neighbors=np.flatnonzero(adj[a]).tolist()
        for task in spec['action_tasks']:
            result[task].append(dict(id=f'{task}:{anchor_index}',anchor=anchor_index,task=task,a=a,g=g,
                options=list(map(str,neighbors)),answers=[str(b) for b in neighbors if dist[b,g]<dist[a,g]]))
        candidates=[b for b in range(len(adj)) if b not in (a,g) and dist[b,g]!=dist[a,g]]
        b=int(rng.choice(candidates))
        for swap,(aa,bb) in enumerate(((a,b),(b,a))):
            result['distance_compare'].append(dict(id=f'distance:{anchor_index}:{swap}',anchor=anchor_index,
                task='distance_compare',a=aa,b=bb,g=g,options=['A','B'],answers=['A' if dist[aa,g]<dist[bb,g] else 'B']))
        for b in neighbors:
            answer='更近' if dist[b,g]<dist[a,g] else '更远' if dist[b,g]>dist[a,g] else '相同'
            result['transition_compare'].append(dict(id=f'transition:{anchor_index}:{b}',anchor=anchor_index,
                task='transition_compare',a=a,b=b,g=g,options=['更近','相同','更远'],answers=[answer]))
    perm=rng.permutation(len(adj))
    while np.any(perm==np.arange(len(adj))):
        perm=rng.permutation(len(adj))
    return result,perm.tolist()


def messages(item,condition,adjacency,reason=None):
    task=item['task'];latent=condition.endswith('_latent');roadmap=condition!='text'
    roles=['a','g'] if task.startswith('neighbor') else ['g','a','b']
    if latent:
        text='图为固定的无向无权图。以下向量表示该图中的节点。\n'
    else:
        text='图为固定的无向无权图，每步沿一条边移动。\n邻接表：\n'+adjacency+'\n'
        text+='\n'.join(f'{role.upper()} 节点编号：{item[role]}' for role in roles)+'\n'
    if roadmap:
        text+='\n节点的 roadmap 表示：\n'+'\n'.join(f'{role.upper()}：<|diag_{role}|>' for role in roles)+'\n'
    if task.startswith('neighbor'):
        text+='A 是当前位置，G 是目标。请选择一个到 G 的最短距离严格小于 A 的合法邻居。\n合法邻居候选：'+', '.join(item['options'])+'。\n'
        instruction='只输出一个候选节点编号，不要解释。'
    elif task=='distance_compare':
        text+='目标为 G。从 A 和 B 分别沿图中的边到达 G，哪个位置需要的最少步数更少？两者距离不同。\n'
        instruction='只输出 A 或 B，不要解释。'
    else:
        text+='目标为 G。已沿一条边从 A 移到 B，到 G 的最短距离变得更近、相同还是更远？\n'
        instruction='只输出 更近、相同 或 更远，不要解释。'
    if task=='neighbor_reason' and reason is None:
        instruction='请在有限篇幅内分析候选方向，不要展开冗长的逐节点搜索。随后会单独要求你给出最终节点编号。'
    conversation=[dict(role='system',content='根据提供的图与状态回答问题。'),dict(role='user',content=text+instruction)]
    if reason is not None:
        conversation += [dict(role='assistant',content=reason),dict(role='user',content='分析阶段结束。现在必须从候选 '+', '.join(item['options'])+' 中选择一个下一节点，只输出编号。')]
    return conversation


class ChoiceGrammar:
    def __init__(self,tokenizer,options):
        self.eos=tokenizer.eos_token_id
        self.pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos
        self.tables=[];self.max_tokens=0
        for choices in options:
            table={}
            for choice in choices:
                seq=tokenizer.encode(choice,add_special_tokens=False)+[self.eos]
                self.max_tokens=max(self.max_tokens,len(seq))
                for i,token in enumerate(seq):table.setdefault(tuple(seq[:i]),set()).add(token)
            self.tables.append(table)
    def __call__(self,batch_id,ids):
        prefix=tuple(ids.tolist())
        return [self.pad] if self.eos in prefix else sorted(self.tables[batch_id][prefix])


class Interface:
    def __init__(self,model,tokenizer,q,rms,adjacency,permutation):
        self.model=model.eval().requires_grad_(False);self.tokenizer=tokenizer
        self.device=model.device;self.dtype=model.get_input_embeddings().weight.dtype
        self.q=torch.tensor(q/rms,dtype=torch.float32,device=self.device)
        self.adjacency=adjacency;self.permutation=permutation
        tokenizer.add_tokens([f'<|diag_{r}|>' for r in ('a','b','g')],special_tokens=True)
        self.slots={r:tokenizer.convert_tokens_to_ids(f'<|diag_{r}|>') for r in ('a','b','g')}
        self.eos=tokenizer.eos_token_id;self.pad=tokenizer.pad_token_id or self.eos
    @torch.no_grad()
    def generate(self,items,condition,adapter,reasons=None,analysis=False,reason_tokens=512):
        sequences=[self.tokenizer.apply_chat_template(messages(x,condition,self.adjacency,None if reasons is None else reasons[i]),
            tokenize=True,add_generation_prompt=True,return_dict=False) for i,x in enumerate(items)]
        width=max(map(len,sequences));ids=torch.full((len(items),width),self.pad,dtype=torch.long,device=self.device);mask=torch.zeros_like(ids)
        for i,seq in enumerate(sequences):ids[i,-len(seq):]=torch.tensor(seq,device=self.device);mask[i,-len(seq):]=1
        safe=ids.clone()
        for slot in self.slots.values():safe[safe==slot]=self.pad
        embeds=self.model.get_input_embeddings()(safe)
        if adapter is not None:
            for role,slot in self.slots.items():
                for i,item in enumerate(items):
                    positions=ids[i]==slot
                    if not positions.any():continue
                    assert positions.sum().item()==1
                    node=item[role]
                    if condition.endswith('_mismatch'):node=self.permutation[node]
                    embeds[i,positions]=adapter(self.q[node]).to(self.dtype)
        grammar=None if analysis else ChoiceGrammar(self.tokenizer,[x['options'] for x in items])
        budget=reason_tokens if analysis else grammar.max_tokens
        assert width+budget<=self.model.config.max_position_embeddings
        cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=budget,eos_token_id=self.eos,pad_token_id=self.pad,use_cache=True)
        generated=self.model.generate(inputs_embeds=embeds,attention_mask=mask,generation_config=cfg,prefix_allowed_tokens_fn=grammar)
        out=[]
        for row in generated.tolist():
            ended=self.eos in row;seq=row[:row.index(self.eos)] if ended else row
            out.append(dict(output=self.tokenizer.decode(seq,skip_special_tokens=True,clean_up_tokenization_spaces=False),tokens=len(seq),ended=ended))
        return out


def summarize(rows):
    result=dict(total=len(rows),correct=sum(r['correct'] for r in rows),invalid=sum(not r['valid'] for r in rows))
    result['accuracy']=result['correct']/len(rows)
    groups={label:[r for r in rows if r['answers']==[label]] for label in sorted({r['answers'][0] for r in rows})}
    if rows[0]['task']=='transition_compare':
        result['per_class']={k:dict(total=len(v),correct=sum(x['correct'] for x in v),accuracy=sum(x['correct'] for x in v)/len(v)) for k,v in groups.items()}
        result['macro_accuracy']=float(np.mean([v['accuracy'] for v in result['per_class'].values()]))
        result['confusion']={k:dict(Counter(x['output'] for x in v)) for k,v in groups.items()}
    if rows[0]['task']=='distance_compare':
        anchors={r['anchor'] for r in rows}
        result['swap_consistency']=sum(len({r['output'] for r in rows if r['anchor']==a})==2 for a in anchors)/len(anchors)
    if rows[0]['task']=='neighbor_reason':
        result['analysis_budget_hits']=sum(not r['analysis']['ended'] for r in rows)
    return result


def diagnostic_checkpoint(condition, spec, base):
    """Use configured checkpoint groups; retain the original report-run defaults."""
    group=condition.split('_')[0]
    binding=spec.get('checkpoint_groups',{}).get(group)
    if binding is None:
        binding=dict(adapter=group,checkpoint_condition=group+'_report',checkpoint='selected')
    continuation=Path(spec['continuation_run']) if spec.get('continuation_run') else None
    return binding['adapter'],checkpoint_for_condition(binding,spec,base,continuation)


def reused_items(source, node_count):
    """Read the original probe questions and mismatch mapping without regeneration."""
    assert read(source/'config.json')['node_count']==node_count
    return read(source/'items.json'),read(source/'permutation.json')


def worker(args,spec):
    base=args.base;root=args.output;task=args.task;condition=args.condition
    out=root/f'{task}__{condition}';out.mkdir(exist_ok=False)
    old,runtime=read(base/'config.json'),read(base/'runtime.json')
    with np.load(Path(runtime['asset_dir'])/'map.npz') as data:q=data['q'].copy()
    adj,_=graph_data(base)
    tokenizer=AutoTokenizer.from_pretrained(runtime['model_dir'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(runtime['model_dir'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda')
    interface=Interface(model,tokenizer,q,runtime['q_global_rms'],adjacency_text(adj),read(root/'permutation.json'))
    adapter=None
    checkpoint=None
    if condition!='text':
        kind,checkpoint=diagnostic_checkpoint(condition,spec,base)
        adapter=make_adapter(old,kind,model.config.hidden_size,'cuda')
        adapter.load_state_dict(load_file(str(checkpoint),device='cuda'));adapter.requires_grad_(False)
    items=read(root/'items.json')[task]
    write_json(out/'runtime.json',dict(started_utc=utc(),source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),base=str(base),condition=condition,task=task,checkpoint_path=str(checkpoint) if checkpoint else None,gpu=torch.cuda.get_device_name(),hostname=os.uname().nodename))
    write_json(out/'prompt_example.json',messages(items[0],condition,interface.adjacency))
    rows=[]
    for batch in chunks(items,spec['generation']['batch_size']):
        analysis=interface.generate(batch,condition,adapter,analysis=True,reason_tokens=spec['generation']['reason_tokens']) if task=='neighbor_reason' else None
        answers=interface.generate(batch,condition,adapter,reasons=[r['output'] for r in analysis] if analysis else None)
        for i,(item,answer) in enumerate(zip(batch,answers)):
            row=dict(item,**answer,condition=condition,valid=answer['output'] in item['options'],correct=answer['output'] in item['answers'])
            if analysis:row['analysis']=analysis[i]
            rows.append(row);append_json(out/'predictions.jsonl',row)
        print(json.dumps(dict(task=task,condition=condition,completed=len(rows),total=len(items))),flush=True)
    write_json(out/'summary.json',summarize(rows));write_json(out/'completed.json',dict(completed_utc=utc()))


def queue(args,spec):
    jobs=Queue()
    for task in spec['action_tasks']+spec['probe_tasks']:
        for condition in spec['conditions']+(spec['probe_extra_conditions'] if task in spec['probe_tasks'] else []):jobs.put((task,condition))
    logs=args.output/'logs';logs.mkdir(exist_ok=False)
    def run(gpu):
        while True:
            try:task,condition=jobs.get_nowait()
            except Empty:return
            with (logs/f'{task}__{condition}.log').open('w') as f:
                cmd=[sys.executable,'-u','-m','experiments.cml_map_scaling.src.step2_decision_diagnostics','--mode','worker','--base',str(args.base),'--output',str(args.output),'--task',task,'--condition',condition]
                p=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
                print(json.dumps(dict(task=task,condition=condition,gpu=gpu,pid=p.pid)),flush=True)
                if p.wait():raise RuntimeError(f'{task} {condition} failed')
    write_json(args.output/'queue_status.json',dict(phase='running',started_utc=utc()))
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures=[pool.submit(run,g) for g in args.gpus]
            for future in futures:future.result()
    except Exception as error:
        write_json(args.output/'queue_status.json',dict(phase='failed',error=str(error)));raise
    summaries={p.parent.name:read(p) for p in args.output.glob('*/summary.json')}
    write_json(args.output/'summary.json',summaries)
    write_json(args.output/'queue_status.json',dict(phase='complete',completed_utc=utc(),jobs=len(summaries)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','queue','worker'],required=True)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/cml_map_scaling/configs/step2_decision_diagnostics.json')
    parser.add_argument('--continuation-run',type=Path)
    parser.add_argument('--items-from',type=Path,help='Reuse previously saved probe questions and mismatch mapping')
    parser.add_argument('--task');parser.add_argument('--condition');parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args();spec=read(args.config)
    if args.mode=='prepare':
        if spec.get('require_saved_items') and args.items_from is None:
            raise ValueError('This diagnostic requires --items-from to preserve the original questions')
        node_count=read(args.base/'config.json')['assets']['node_count']
        assert node_count in spec['node_counts']
        spec['node_count']=node_count
        if args.continuation_run:
            continuation_runtime=read(args.continuation_run/'runtime.json')
            assert continuation_runtime.get('completed_utc')
            assert Path(continuation_runtime['base_run']).resolve()==args.base.resolve()
            spec['continuation_run']=str(args.continuation_run)
        if args.items_from:
            items,permutation=reused_items(args.items_from,node_count)
            spec['items_source']=str(args.items_from)
        else:
            items,permutation=make_items(args.base,spec)
        args.output.mkdir(parents=True,exist_ok=False)
        write_json(args.output/'config.json',spec);write_json(args.output/'items.json',items);write_json(args.output/'permutation.json',permutation)
        print({k:len(v) for k,v in items.items()})
    elif args.mode=='worker':worker(args,read(args.output/'config.json'))
    else:queue(args,read(args.output/'config.json'))


if __name__=='__main__':main()
