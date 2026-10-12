"""Evaluate validation and the fixed test set for every arriving epoch model."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

from .evaluate import summarize
from .protocol import atomic_json


def collect_model(root, label, split, modes, counts, size):
    saved = root/label/split/'summary.json'
    if saved.exists():
        return json.loads(saved.read_text())
    summaries = {}
    for mode in modes:
        paths = [root/label/split/mode/f'shard_{i:03d}' for i in range((counts[split]+size-1)//size)]
        if not all((p/'summary.json').exists() for p in paths):
            return None
        cases = [json.loads(line) for p in paths for line in (p/'cases.jsonl').read_text().splitlines()]
        if len(cases) != counts[split] or len({c['trajectory_id'] for c in cases}) != counts[split]:
            raise ValueError('Evaluation shard coverage differs from physical task list')
        summaries[mode] = summarize(cases)
    result = dict(model=label, split=split, metrics=summaries)
    atomic_json(saved,result)
    return result


def pending_jobs(root,models,counts,modes,size):
    result=[]
    for label,model in models:
        for split in ['validation','test']:
            # A complete split can survive a host migration without its large raw shards.
            if (root/label/split/'summary.json').exists():
                continue
            for mode in modes:
                for shard in range((counts[split]+size-1)//size):
                    out=root/label/split/mode/f'shard_{shard:03d}'
                    if not (out/'summary.json').exists(): result.append((label,model,split,mode,shard,out))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    config=json.loads(Path(profile['config']).read_text())
    run=Path(profile['run']); data=Path(profile['data']); root=run/'evaluation'
    root.mkdir(exist_ok=True)
    while not (data/'ready.json').exists(): time.sleep(30)
    counts=Counter(r['split'] for r in json.loads((data/'evaluation.json').read_text()))
    modes=config['evaluation']['epoch_modes']; size=config['evaluation']['shard_tasks']
    active={}; selected=None; failed=False
    def jobs():
        models=[('base',Path(profile['base_model']))]
        models.extend((f'epoch_{i}',run/'models'/f'epoch_{i}') for i in range(1,config['training']['epochs']+1)
                      if (run/'models'/f'epoch_{i}'/'evaluation_ready.json').exists())
        return pending_jobs(root,models,counts,modes,size)
    while True:
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is not None:
                log.close(); del active[gpu]
                if code:
                    atomic_json(run/'evaluation_failure.json',dict(gpu=gpu,job=str(job[-1]),returncode=code))
                    failed=True
        if failed:
            # Other independent shards may finish; no automatic change of protocol or retries.
            if not active: raise RuntimeError('A worker failed; inspect evaluation_failure.json and its worker.log')
            time.sleep(10); continue
        completed={};completed_test={}
        for label in ['base']+[f'epoch_{i}' for i in range(1,config['training']['epochs']+1)]:
            summary=collect_model(root,label,'validation',modes,counts,size)
            if summary: completed[label]=summary
            test_summary=collect_model(root,label,'test',modes,counts,size)
            if test_summary: completed_test[label]=test_summary
        if selected is None and all(f'epoch_{i}' in completed for i in range(1,config['training']['epochs']+1)):
            epoch=max(range(1,config['training']['epochs']+1),key=lambda i:(
                completed[f'epoch_{i}']['metrics']['rollout']['arrival']['rate'],
                completed[f'epoch_{i}']['metrics']['rollout']['arrival_and_stop']['rate'],-i))
            selected=f'epoch_{epoch}'
            atomic_json(run/'selected_checkpoint.json',dict(model=selected,criteria=config['evaluation']['selection'],
                                                          validation=completed[selected]))
        pending=jobs(); occupied={str(job[-1]) for _,job,_ in active.values()}
        pending=[job for job in pending if str(job[-1]) not in occupied]
        for gpu in config['execution']['evaluation_gpus']:
            if gpu in active or not pending: continue
            job=pending.pop(0); label,model,split,mode,shard,out=job
            out.mkdir(parents=True,exist_ok=True); log=(out/'worker.log').open('wb')
            command=[profile['python'],'-u','-m','experiments.llm_task_sft.src.evaluate','--config',profile['config'],
                '--data',str(data),'--model-path',str(model),'--out',str(out),'--split',split,'--mode',mode,'--shard-index',str(shard)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            process=subprocess.Popen(command,cwd=profile['code'],env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(process,job,log)
        base_test=completed_test.get('base')
        selected_test=completed_test.get(selected) if selected else None
        state=dict(active=[dict(gpu=gpu,pid=p.pid,job=str(job[-1])) for gpu,(p,job,_) in active.items()],
                   completed_validation=list(completed),completed_test=list(completed_test),
                   selected=selected,pending_shards=len(pending),utc_epoch=time.time())
        atomic_json(run/'evaluation_status.json',state)
        all_test=len(completed_test)==config['training']['epochs']+1
        if selected_test and base_test and 'base' in completed and all_test and not active:
            atomic_json(run/'summary.json',dict(validation=completed,test=completed_test,
                base_test=base_test,selected_test=selected_test,selected=selected))
            return
        time.sleep(10)


if __name__=='__main__': main()
