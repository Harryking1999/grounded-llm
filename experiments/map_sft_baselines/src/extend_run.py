"""Continue one completed training phase with its full optimizer and RNG state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import atomic_json
from .contract import training_steps
from .train import validate_resume_contract


def main():
    parser=argparse.ArgumentParser()
    for name in ('config','root','data','model-path'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--previous-step',required=True,type=int)
    parser.add_argument('--gpus',required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    root=args.root.resolve();out=root/'training'/args.baseline
    gpus=[int(value) for value in args.gpus.split(',')]
    if len(gpus)!=cfg['training']['world_size'] or len(set(gpus))!=len(gpus):
        raise ValueError('GPU allocation differs from the configured training topology')
    status=root/'extension_status.json'
    atomic_json(status,dict(status='waiting_for_previous_phase',previous_step=args.previous_step,
                           target_updates=training_steps(cfg),pid=os.getpid()))
    while True:
        queue=root/'training_queue.json'
        queue=json.loads(queue.read_text()) if queue.exists() else {}
        if queue.get('failed'):
            raise RuntimeError('Previous phase failed; automatic training retry is forbidden')
        finished=out/'finished.json'
        done=json.loads(finished.read_text()) if finished.exists() else None
        if done:
            if done.get('status')!='completed' or done.get('step')!=args.previous_step:
                raise RuntimeError('Previous phase did not complete the expected update budget')
            # Wait for the old launcher to exit and its training queue to publish completion.
            if queue.get('status')=='finished' and not queue.get('active') and not queue.get('pending'):
                break
        time.sleep(60)
    checkpoint=out/f'checkpoint-{args.previous_step}'
    state=json.loads((checkpoint/'state.json').read_text())
    validate_resume_contract(state,cfg,args.baseline,allow_horizon_extension=True)
    if state['step']!=args.previous_step or not (checkpoint/'resume.pt').is_file():
        raise ValueError('Complete training resume state is unavailable')
    while any(int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
              '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=1000
              for gpu in gpus):
        time.sleep(60)
    atomic_json(root/'previous_phase_finished.json',done)
    finished.unlink()
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone',
             f"--nproc-per-node={cfg['training']['world_size']}",'-m',
             'experiments.map_sft_baselines.src.train','--config',str(args.config),
             '--data',str(args.data),'--model-path',str(args.model_path),'--out',str(out),
             '--baseline',args.baseline,'--resume',str(checkpoint),'--extend-training']
    with (out/'extension_launcher.log').open('a') as log:
        process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus),
                                 stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        record=dict(status='running',previous_step=args.previous_step,target_updates=training_steps(cfg),
                    pid=os.getpid(),training_pid=process.pid,command=command,started=time.time())
        atomic_json(status,record)
        atomic_json(root/'training_queue.json',dict(status='running',active={'extension':dict(
            pid=process.pid,baseline=args.baseline,gpus=gpus)},pending=[],failed=[]))
        result=process.wait()
    completed=json.loads(finished.read_text()) if finished.exists() else {}
    success=(result==0 and completed.get('status')=='completed' and
             completed.get('step')==training_steps(cfg))
    record.update(status='completed' if success else 'failed',returncode=result,finished=time.time())
    atomic_json(status,record)
    atomic_json(root/'training_queue.json',dict(status=record['status'],active={},pending=[],
                failed=[] if success else [dict(stage='extension',returncode=result)]))
    if not success:
        raise RuntimeError('Extended training failed; inspect extension_launcher.log without automatic retry')


if __name__=='__main__':
    main()
