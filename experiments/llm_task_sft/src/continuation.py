"""Train another half epoch, evaluate only 510 fixed tasks, and repeat."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .epoch_queue import collect_model, pending_jobs
from grounded_llm.core.files import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def cycle_steps(contract,config):
    steps_epoch=config['data']['samples_per_epoch']//config['training']['global_batch_size']
    interval=steps_epoch*contract['save_and_evaluate_every_epochs']
    if interval != int(interval) or interval <= 0:
        raise ValueError('Checkpoint interval must be a positive whole number of updates')
    if contract['evaluation']['splits'] != ['test'] or contract['evaluation']['physical_tasks'] != 510:
        raise ValueError('This continuation is restricted to the fixed 510-task monitor')
    if contract['epoch_limit'] is not None or contract['learning_rate_policy'] != 'hold_original_cosine_terminal_rate':
        raise ValueError('Unexpected continuation horizon or LR policy')
    return steps_epoch,int(interval)


def record_point(out,epoch,step,checkpoint,summary):
    path=out/'curve.json'
    curve=read(path) if path.exists() else dict(split='test',tasks=510,use='training_monitor_only',points=[])
    points={p['step']:p for p in curve['points']}
    points[step]=dict(epoch=epoch,step=step,model_path=str(checkpoint),metrics=summary['metrics'])
    curve['points']=[points[k] for k in sorted(points)]
    best=max(curve['points'],key=lambda p:(
        p['metrics']['rollout']['arrival']['rate'],p['metrics']['rollout']['arrival_and_stop']['rate'],-p['epoch']))
    curve['best_observed_epoch']=best['epoch']
    curve['best_observed_model_path']=best['model_path']
    atomic_json(path,curve)


def train_segment(profile,config_path,config,checkpoint,target,out,status):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,profile['training_gpus'])),
             OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    command=[profile['python'],'-u','-m','torch.distributed.run','--standalone',
        f"--nproc-per-node={config['training']['world_size']}",'-m','experiments.llm_task_sft.src.train',
        '--config',str(config_path),'--data',profile['data'],'--model-path',profile['base_model'],
        '--out',str(out/'training'),'--resume',str(checkpoint),'--stop-step',str(target)]
    log_path=out/'logs'/f'train_to_{target}.log'
    with log_path.open('ab') as log:
        process=subprocess.Popen(command,cwd=profile['code'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status('training',child_pid=process.pid,target_step=target,resume=str(checkpoint),log=str(log_path))
        result=process.wait()
    if result: raise RuntimeError(f'Training returned {result}; inspect {log_path}')
    checkpoint=out/'training'/f'checkpoint-{target}'
    if read(checkpoint/'evaluation_ready.json')['step'] != target:
        raise RuntimeError('Training did not publish the requested complete checkpoint')
    return checkpoint


def evaluate_checkpoint(profile,config_path,config,contract,checkpoint,out,status):
    metadata=read(checkpoint/'evaluation_ready.json')
    label=f"epoch_{metadata['epoch']:g}"
    root=out/'evaluation'
    counts=Counter(x['split'] for x in read(Path(profile['data'])/'evaluation.json'))
    if counts['test'] != 510: raise ValueError('Evaluation data is not the fixed 510-task set')
    modes=contract['evaluation']['modes'];size=config['evaluation']['shard_tasks']
    active={};failures=[]
    while True:
        for gpu,(process,job,log) in list(active.items()):
            result=process.poll()
            if result is not None:
                log.close();del active[gpu]
                if result: failures.append(dict(job=str(job[-1]),returncode=result))
        if failures:
            if not active: raise RuntimeError(f'Evaluation workers failed: {failures}')
            time.sleep(5);continue
        summary=collect_model(root,label,'test',modes,counts,size)
        if summary and not active: return summary
        pending=pending_jobs(root,[(label,checkpoint)],counts,modes,size,splits=contract['evaluation']['splits'])
        occupied={str(job[-1]) for _,job,_ in active.values()}
        pending=[job for job in pending if str(job[-1]) not in occupied]
        for gpu in profile['evaluation_gpus']:
            if gpu in active or not pending: continue
            job=pending.pop(0);_,model,split,mode,shard,destination=job
            destination.mkdir(parents=True,exist_ok=True);log=(destination/'worker.log').open('ab')
            command=[profile['python'],'-u','-m','experiments.llm_task_sft.src.evaluate',
                '--config',str(config_path),'--data',profile['data'],'--model-path',str(model),
                '--out',str(destination),'--split',split,'--mode',mode,'--shard-index',str(shard)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            process=subprocess.Popen(command,cwd=profile['code'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(process,job,log)
        status('evaluating',epoch=metadata['epoch'],checkpoint=str(checkpoint),pending_shards=len(pending),
            active=[dict(gpu=gpu,pid=p.pid,job=str(job[-1])) for gpu,(p,job,_) in active.items()])
        time.sleep(5)


def run_cycle(profile,config_path,config,contract,checkpoint,target,out,status):
    ready=out/'training'/f'checkpoint-{target}'/'evaluation_ready.json'
    if not ready.exists():
        checkpoint=train_segment(profile,config_path,config,checkpoint,target,out,status)
    else:
        checkpoint=ready.parent
    summary=evaluate_checkpoint(profile,config_path,config,contract,checkpoint,out,status)
    record_point(out,read(checkpoint/'evaluation_ready.json')['epoch'],target,checkpoint,summary)
    status('cycle_complete',step=target,checkpoint=str(checkpoint))
    return checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    args=parser.parse_args();profile=read(args.profile)
    contract_path=Path(profile['contract']);contract=read(contract_path)
    profile=dict(profile,training_gpus=contract['execution']['training_gpus'],
                 evaluation_gpus=contract['execution']['evaluation_gpus'])
    config_path=contract_path.parent/contract['base_config'];config=read(config_path)
    steps_epoch,interval=cycle_steps(contract,config)
    out=Path(profile['out']);(out/'logs').mkdir(parents=True,exist_ok=True)
    # A single local coordinator owns the sequential train/evaluate cycle.
    import fcntl
    with (out/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(phase,**details):
            atomic_json(out/'status.json',dict(phase=phase,pid=os.getpid(),utc_epoch=time.time(),**details))
        try:
            baseline=Path(profile['baseline_run'])/'summary.json'
            while not baseline.exists():
                status('waiting_initial_evaluation',summary=str(baseline));time.sleep(30)
            for label,summary in read(baseline)['test'].items():
                epoch=0 if label=='base' else float(label.split('_',1)[1])
                model=Path(profile['base_model']) if label=='base' else Path(profile['baseline_run'])/'models'/label
                record_point(out,epoch,round(epoch*steps_epoch),model,summary)
            checkpoint=Path(profile['resume'])
            if read(checkpoint/'evaluation_ready.json')['step'] != contract['resume_step']:
                raise ValueError('Continuation must start from the original third-epoch state')
            later=[p.parent for p in (out/'training').glob('checkpoint-*/evaluation_ready.json')]
            if later:
                checkpoint=max(later,key=lambda p:read(p/'evaluation_ready.json')['step'])
                summary=evaluate_checkpoint(profile,config_path,config,contract,checkpoint,out,status)
                marker=read(checkpoint/'evaluation_ready.json')
                record_point(out,marker['epoch'],marker['step'],checkpoint,summary)
            while not (out/'stop_after_cycle').exists():
                step=read(checkpoint/'evaluation_ready.json')['step']
                free=shutil.disk_usage(out).free/2**30
                if free < contract['execution']['minimum_free_disk_gib']:
                    status('disk_space_required',free_gib=free,step=step);return
                checkpoint=run_cycle(profile,config_path,config,contract,checkpoint,step+interval,out,status)
            status('stopped_by_request',checkpoint=str(checkpoint))
        except Exception as error:
            status('failed',error=str(error));raise


if __name__=='__main__': main()
