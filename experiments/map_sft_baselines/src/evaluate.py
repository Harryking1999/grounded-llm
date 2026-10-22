"""Identical reference states and own-history rollout with existing NDCG gains."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import torch
from transformers import AutoTokenizer,StoppingCriteria,StoppingCriteriaList
from grounded_llm.tasks.blocks.protocol import parse_control,action_from_key,legal_actions,action_key
from grounded_llm.tasks.blocks.oracle import DistanceOracle
from grounded_llm.core.reader.batching import to_device
from grounded_llm.core.reader.batching import timeline_maps
from grounded_llm.tasks.blocks.evaluation import score
from experiments.flamingo_map_reader.src.counterfactual import reverse_candidate_q
from grounded_llm.tasks.blocks.qmap import FrozenBoardMap
import grounded_llm.tasks.blocks.rules as rules
from grounded_llm.tasks.blocks.sft import prompt,sorted_step,pseudo_timeline,atomic_json
from grounded_llm.core.qmap.reader import CachedQMap as CachedMap
from .modeling import build
from .metrics import summarize

class Boundary(StoppingCriteria):
    def __init__(self,tokenizer,prefix):self.tokenizer,self.prefix=tokenizer,prefix
    def __call__(self,input_ids,scores,**kwargs):
        text=self.tokenizer.decode(input_ids[0,self.prefix:],skip_special_tokens=True)
        return '</action>' in text or '<done/>' in text


def main():
    p=argparse.ArgumentParser()
    for name in ('config','data','model-path','q-checkpoint','checkpoint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--mode',choices=['TF','RO','probe'],required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int)
    a=p.parse_args();cfg=json.loads(a.config.read_text());spec=cfg['baselines'][a.baseline];torch.set_num_threads(1)
    if cfg.get('task') == 'multitask':
        from .multitask_evaluate import evaluate
        return evaluate(a, cfg)
    if cfg.get('task') in ('cube2', 'puzzle8', 'path'):
        from .restoration_evaluate import evaluate
        return evaluate(a, cfg)
    records=json.loads((a.data/'evaluation.json').read_text())
    if len(records)!=cfg['evaluation']['tasks']:raise ValueError('Evaluation cohort differs from configuration')
    donor_records=records
    if a.mode=='probe':records=[records[i] for i in json.loads((a.data/'probe_indices.json').read_text())]
    training_probe=[]
    if a.mode=='probe' and cfg['evaluation'].get('first_turn_diagnosis',False):
        training_probe=json.loads((a.data/'training_probe.json').read_text())
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
                update=prompt(state,goal,executed,first=not messages,start=int(record['start']),with_map=spec['real_map'],response_order=spec.get('response_order','reason_first'))
                messages.append(dict(role='user',content=update));steps.append(step)
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                map_ids=previous_ids+[len(steps)-1]*(len(ids)-len(previous_ids))
                text=generate(messages,steps,map_ids);row=score(text,step,oracle,spec['real_map']);turns.append(row)
                if a.mode=='probe' and not step.done:
                    altered,_=reverse_candidate_q(step);other=generate(messages,[altered],[0]*len(ids));row['reverse']=score(other,step,oracle,True);row['changed_action']=row['candidate']!=row['reverse']['candidate']
                    if index==0 and cfg['evaluation'].get('first_turn_diagnosis',False):
                        from experiments.flamingo_map_reader.src.trajectory_protocol import numbered_step
                        from grounded_llm.tasks.blocks.prompts import turn_prompt
                        zero=replace(step,map_batch=replace(step.map_batch,vectors=torch.zeros_like(step.map_batch.vectors)))
                        zero_score=score(generate(messages,[zero],[0]*len(ids)),step,oracle,True)
                        row['map_zero']=zero_score
                        row['map_zero_action_changed']=(zero_score['candidate'],zero_score['error'])!=(row['candidate'],row['error'])
                        numbered=numbered_step(step,tuple(reversed(range(len(step.candidate_actions)))))
                        new_text=update.split('[Environment update]',1)[0]+turn_prompt(numbered,[])
                        new_messages=[dict(role='user',content=new_text)]
                        new_ids=tokenizer.apply_chat_template(new_messages,tokenize=True,add_generation_prompt=True)
                        result=score(generate(new_messages,[numbered],[0]*len(new_ids)),numbered,oracle,True)
                        row['renumber']=result
                        row['renumber_safe']=result['safe']
                        def physical(scored,source):
                            return source.candidate_actions[scored['candidate']-1] if scored['candidate'] is not None and scored['error'] is None else None
                        row['renumber_same_physical_action']=physical(result,numbered)==physical(row,step) and physical(row,step) is not None
                        donor=next((r for r in donor_records if int(r['start'])!=state and
                            len(legal_actions(int(r['start'])))==len(step.candidate_actions)),None)
                        row['text_swap_available']=donor is not None
                        if donor:
                            donor_text=prompt(int(donor['start']),int(donor['goal']),[],True,int(donor['start']),True,spec['response_order'])
                            swapped=[dict(role='user',content=donor_text)]
                            swapped_ids=tokenizer.apply_chat_template(swapped,tokenize=True,add_generation_prompt=True)
                            result=score(generate(swapped,[step],[0]*len(swapped_ids)),step,oracle,True)
                            row['text_swap']=dict(result,donor_id=donor['trajectory_id'])
                            row['text_swap_action_changed']=(result['candidate'],result['error'])!=(row['candidate'],row['error'])
                        probe_index=a.start+len(cases)
                        if probe_index<len(training_probe):
                            train=training_probe[probe_index]
                            train_step=sorted_step(qmap,int(train['start']),int(train['goal']))
                            train_text=prompt(int(train['start']),int(train['goal']),[],True,int(train['start']),True,spec['response_order'])
                            train_messages=[dict(role='user',content=train_text)]
                            train_ids=tokenizer.apply_chat_template(train_messages,tokenize=True,add_generation_prompt=True)
                            row['training_probe']=dict(score(generate(train_messages,[train_step],[0]*len(train_ids)),train_step,oracle,True),trajectory_id=train['trajectory_id'])
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
