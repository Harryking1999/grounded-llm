"""Independent training lanes and latest-first four-GPU evaluation workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from grounded_llm.core.files import atomic_json
from .metrics import summarize
from .contract import steps_per_epoch,checkpoint_steps,cohort_size,evaluation_modes,checkpoint_priority

def evaluation_jobs(root,config,records=None):
    jobs=[]
    if records is None:
        records=[json.loads(marker.read_text()) for marker in Path(root).glob('models/B*/checkpoint-*/evaluation_ready.json')]
    for meta in records:
        base=meta['baseline'];step=meta['step']
        if base in config.get('execution',{}).get('retired_baselines',[]):continue
        checkpoint=Path(root)/'models'/base/f'checkpoint-{step}'
        modes=evaluation_modes(config,base)
        for mode in modes:
            total=cohort_size(config,mode)
            for start in range(0,total,config['evaluation']['shard_tasks']):
                stop=min(start+config['evaluation']['shard_tasks'],total);out=Path(root)/'evaluation'/base/f'checkpoint-{step}'/f'{mode}_{start:04d}_{stop:04d}'
                if not (out/'summary.json').exists():jobs.append((step,meta['epoch'],base,mode,start,stop,checkpoint,out))
    return sorted(jobs,key=lambda j:(checkpoint_priority(j[0],config),
        {'RO':0,'TF':1,'probe':2}[j[3]],j[2],j[4]))

def collect(root,config):
    for base in config['baselines']:
        curve=[]
        for checkpoint in sorted((Path(root)/'evaluation'/base).glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])):
            modes=evaluation_modes(config,base)
            all_metrics={};valid=True
            for mode in modes:
                cases=[];total=cohort_size(config,mode)
                for start in range(0,total,config['evaluation']['shard_tasks']):
                    stop=min(start+config['evaluation']['shard_tasks'],total);path=checkpoint/f'{mode}_{start:04d}_{stop:04d}'
                    if not (path/'summary.json').exists():valid=False;break
                    cases.extend(json.loads(line) for line in (path/'cases.jsonl').read_text().splitlines())
                if not valid:break
                if len(cases)!=total or len({c['trajectory_id'] for c in cases})!=total:raise ValueError('Evaluation missing/duplicate IDs')
                all_metrics[mode]=summarize(cases)
            if valid:
                step=int(checkpoint.name.split('-')[-1]);row=dict(step=step,epoch=step/steps_per_epoch(config),metrics=all_metrics)
                atomic_json(checkpoint/'summary.json',row);curve.append(row)
        if curve:
            destination=Path(root)/'results'/base
            atomic_json(destination/'curve.json',curve)
            columns=['到达目标','保持可达 TF／RO']
            names=['safe']
            if config['baselines'][base]['real_map']:
                columns+=['地图最优 TF／RO','位置 NDCG@1 TF／RO','位置 NDCG@10 TF／RO','距离 NDCG@1 TF／RO','距离 NDCG@10 TF／RO']
                names+=['map_optimal','ndcg_inverse_rank_at_1','ndcg_inverse_rank_at_10','ndcg_minmax_distance_at_1','ndcg_minmax_distance_at_10']
            else:columns+=['单步最优率 TF／RO'];names+=['optimal']
            lines=[f'# {base} 半轮评测','', '| epoch | '+' | '.join(columns)+' |','|---|'+'---|'*len(columns)]
            lookup={r['step']:r for r in curve}
            def percent(x):return '—' if x is None else f'{100*x:.2f}%'
            tasks=cohort_size(config,'RO')
            for step in checkpoint_steps(config):
                row=lookup.get(step)
                if row:
                    metrics=row['metrics'];cells=[f"{metrics['RO']['arrival']}/{tasks}（{percent(metrics['RO']['arrival']/tasks)}）"]
                    for name in names:cells.append('／'.join(percent(metrics[mode][name]['value']) for mode in ['TF','RO']))
                else:cells=['—']*len(columns)
                lines.append(f'| {step/steps_per_epoch(config):g} | '+' | '.join(cells)+' |')
            (destination/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('role',choices=['train','evaluate'])
    for name in ('config','root','data','model-path'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--q-checkpoint',type=Path)
    p.add_argument('--node-id',default='local')
    p.add_argument('--gpus',help='Node-specific evaluation GPUs; does not alter the study config')
    p.add_argument('--coordination-profile',type=Path,help='Ignored runtime SSH profile for the common shard authority')
    p.add_argument('--resume-workers',action='store_true')
    p.add_argument('--training-queue',type=Path,help='Use only released training GPUs, after pending training jobs are assigned')
    p.add_argument('--after-training',type=Path,help='Wait for this training finished.json to report completed before evaluating')
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.root.mkdir(parents=True,exist_ok=True)
    if a.role=='evaluate':
        from .evaluation_queue import run
        return run(a,cfg)
    active={};failed=[]
    def free_gpu(gpu):
        used=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        return int(used.strip())<1000
    if a.role=='train':pending=list(cfg['execution']['initial_baselines'])+list(cfg['execution']['pending_baselines'])
    if a.role=='train' and a.resume_workers:
        from .evaluation_queue import ExistingWorker
        status=a.root/'training_queue.json'
        previous=json.loads(status.read_text()) if status.exists() else {}
        pending=[base for base in pending if not (a.root/'training'/base/'finished.json').exists()]
        for lane,record in previous.get('active',{}).items():
            base=record['baseline'];pid=record['pid'];out=a.root/'training'/base
            path=Path(f'/proc/{pid}/cmdline')
            argv=path.read_bytes().decode().split('\0') if path.exists() else []
            if 'torch.distributed.run' not in argv or '--out' not in argv or argv[argv.index('--out')+1]!=str(out):continue
            if base not in pending:raise RuntimeError('Unexpected live training lane; wait for requested stop')
            if record['gpus']!=cfg['execution']['gpu_pairs'][int(lane)]:raise RuntimeError('Live training GPU assignment changed')
            active[int(lane)]=(ExistingWorker(pid,out,completion_marker='finished.json'),base,None)
            pending.remove(base)
    while True:
        for lane,item in list(active.items()):
            process,job,handle=item
            code=process.poll()
            if code is None:continue
            if handle:handle.close()
            del active[lane]
            if code:failed.append(dict(job=str(job),code=code))
        if a.role=='train':
            if not (a.data/'ready.json').exists():
                atomic_json(a.root/'training_queue.json',dict(status='waiting_for_common_data',
                    active={str(k):dict(pid=v[0].pid,baseline=v[1],gpus=cfg['execution']['gpu_pairs'][k]) for k,v in active.items()},
                    pending=pending,failed=failed));time.sleep(30);continue
            for lane,pair in enumerate(cfg['execution']['gpu_pairs']):
                if lane in active or not pending or not all(free_gpu(gpu) for gpu in pair):continue
                base=pending.pop(0);out=a.root/'training'/base;out.mkdir(parents=True,exist_ok=True)
                log=(out/'launcher.log').open('a')
                if len(pair)!=cfg['training']['world_size']:raise ValueError('GPU lane size differs from world size')
                command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',f"--nproc-per-node={cfg['training']['world_size']}",'-m','experiments.map_sft_baselines.src.train','--config',str(a.config),'--data',str(a.data),'--model-path',str(a.model_path),'--out',str(out),'--baseline',base]
                proc=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,pair))),stdout=log,stderr=subprocess.STDOUT)
                active[lane]=(proc,base,log)
            atomic_json(a.root/'training_queue.json',dict(status='running' if active else 'finished',active={str(k):dict(pid=v[0].pid,baseline=v[1],gpus=cfg['execution']['gpu_pairs'][k]) for k,v in active.items()},pending=pending,failed=failed))
            if not active and not pending:break
        time.sleep(cfg['execution']['poll_seconds'])

if __name__=='__main__':main()
