"""Identical reference states and own-history rollout with existing NDCG gains."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,StoppingCriteria,StoppingCriteriaList
from experiments.llm_task_sft.src.protocol import parse_control,action_from_key,legal_actions,action_key
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.flamingo_map_reader.src.train import to_device
from experiments.flamingo_map_reader.src.graph import timeline_maps
from experiments.flamingo_map_reader.src.relations import parse_ranking
from experiments.flamingo_map_reader.src.analyze_blocks_ranking import meeting_ranking_scores
from experiments.flamingo_map_reader.src.counterfactual import reverse_candidate_q
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap
from experiments.gcml_counterexamples.src import blocks as rules
from .common import prompt,sorted_step,pseudo_timeline,atomic_json
from .prepare import CachedMap
from .modeling import build

class Boundary(StoppingCriteria):
    def __init__(self,tokenizer,prefix):self.tokenizer,self.prefix=tokenizer,prefix
    def __call__(self,input_ids,scores,**kwargs):
        text=self.tokenizer.decode(input_ids[0,self.prefix:],skip_special_tokens=True)
        return '</action>' in text or '<done/>' in text

def score(text,step,oracle,with_map):
    before=oracle.distance(step.current,step.goal);candidate,error=parse_control(text,len(step.candidate_actions))
    row=dict(current=str(step.current),goal=str(step.goal),answer=text,candidate=candidate,error=error,live=before>0,rank_eligible=not step.done and bool(step.candidate_actions),safe=False,optimal=False,map_optimal=False)
    if candidate is not None and error is None:
        _,after=step.execute(candidate);remaining=oracle.distance(after,step.goal)
        row.update(after=str(after),safe=before>0 and remaining>=0,optimal=before>0 and remaining==before-1,map_optimal=candidate in step.map_minimal_candidates)
    if with_map and row['rank_eligible']:
        row.update(meeting_ranking_scores(step.candidate_map_distances,parse_ranking(text,len(step.candidate_actions),10)))
    return row

def summarize(cases):
    turns=[r for c in cases for r in c['turns']];live=[r for r in turns if r['live']];ranked=[r for r in turns if r['rank_eligible']]
    def rate(key,rows):return dict(numerator=sum(int(r.get(key,False)) for r in rows),denominator=len(rows),value=sum(int(r.get(key,False)) for r in rows)/len(rows) if rows else None)
    result=dict(tasks=len(cases),arrival=sum(c['arrived'] for c in cases),arrival_and_stop=sum(c['arrived_and_stopped'] for c in cases),safe=rate('safe',live),optimal=rate('optimal',live),map_optimal=rate('map_optimal',ranked))
    for key in ('ndcg_inverse_rank_at_1','ndcg_inverse_rank_at_10','ndcg_minmax_distance_at_1','ndcg_minmax_distance_at_10'):
        rows=[r for r in ranked if key in r];result[key]=dict(denominator=len(rows),value=sum(r[key] for r in rows)/len(rows) if rows else None)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ('config','data','model-path','q-checkpoint','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--mode',choices=['TF','RO','probe'],required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=510)
    a=p.parse_args();cfg=json.loads(a.config.read_text());spec=cfg['baselines'][a.baseline];torch.set_num_threads(1)
    records=json.loads((a.data/'evaluation.json').read_text())
    if a.mode=='probe':records=[records[i] for i in json.loads((a.data/'probe_indices.json').read_text())]
    records=records[a.start:a.stop];tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    reader=build(cfg,a.baseline,a.model_path,False,a.checkpoint).cuda();qmap=CachedMap(FrozenBoardMap.load(a.q_checkpoint))
    oracle=DistanceOracle(cache_limit=2000000,seconds=86400);cases=[];a.out.mkdir(parents=True,exist_ok=True)
    def generate(messages,steps,map_ids):
        prefix=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
        if len(ids)+cfg['evaluation']['max_new_tokens']>cfg['maximum_sequence_tokens']:return '<context_overflow/>'
        kwargs=dict(input_ids=torch.tensor([ids],device='cuda'),attention_mask=torch.ones((1,len(ids)),device='cuda',dtype=torch.long),max_new_tokens=cfg['evaluation']['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([Boundary(tokenizer,len(ids))]))
        if spec['interface']:
            timeline=timeline_maps(steps,map_ids)
            if not spec['real_map']:timeline=pseudo_timeline(timeline)
            kwargs['map_batch']=to_device(timeline,torch.device('cuda'))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=reader.generate(**kwargs)
        return tokenizer.decode(output[0,len(ids):],skip_special_tokens=True).strip()
    with (a.out/'cases.jsonl').open('w') as f:
        for record in records:
            state,goal=int(record['start']),int(record['goal']);executed=[];turns=[];messages=[];steps=[];previous_ids=[];arrived=False;stopped=False
            limit=len(record['reference_actions']) if a.mode!='RO' else cfg['evaluation']['action_budget']
            for index in range(limit+1):
                step=sorted_step(qmap,state,goal)
                if state!=goal and not step.candidate_actions:break
                if a.mode!='RO':messages=[];steps=[];previous_ids=[]
                update=prompt(state,goal,executed,first=not messages,start=int(record['start']),with_map=spec['real_map'])
                messages.append(dict(role='user',content=update));steps.append(step)
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                map_ids=previous_ids+[len(steps)-1]*(len(ids)-len(previous_ids))
                text=generate(messages,steps,map_ids);row=score(text,step,oracle,spec['real_map']);turns.append(row)
                if a.mode=='probe' and not step.done:
                    altered,_=reverse_candidate_q(step);other=generate(messages,[altered],[0]*len(ids));row['reverse']=score(other,step,oracle,True);row['changed_action']=row['candidate']!=row['reverse']['candidate']
                if a.mode!='RO':
                    if index==len(record['reference_actions']):break
                    action=action_from_key(record['reference_actions'][index]);executed.append(action);state=rules.apply(state,action);continue
                if step.done:arrived=True;stopped=row['candidate'] is None and row['error'] is None;break
                if row['error'] or row['candidate'] is None or index==limit:break
                action=legal_actions(state)[row['candidate']-1];executed.append(action);state=int(row['after']);arrived|=state==goal
                messages.append(dict(role='assistant',content=text))
                complete=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
                previous_ids=map_ids+[len(steps)-1]*(len(complete)-len(map_ids))
            case=dict(trajectory_id=record['trajectory_id'],arrived=arrived,arrived_and_stopped=arrived and stopped,turns=turns)
            f.write(json.dumps(case)+'\n');f.flush();cases.append(case)
            print(json.dumps(dict(done=len(cases),total=len(records))),flush=True)
    atomic_json(a.out/'summary.json',dict(mode=a.mode,baseline=a.baseline,checkpoint=json.loads((a.checkpoint/'evaluation_ready.json').read_text()),metrics=summarize(cases)))

if __name__=='__main__':main()
