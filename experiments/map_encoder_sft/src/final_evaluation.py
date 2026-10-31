"""Give both study evaluation nodes to finished-endpoint OOD before ID.

Reuse the existing evaluator, collector and shard ownership protocol. Training
continues throughout. Credentials arrive via stdin.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

from filelock import FileLock
from grounded_llm.core.files import atomic_json
from experiments.map_sft_baselines.src.evaluation_queue import ExistingWorker, existing_jobs
from experiments.map_sft_baselines.src.shard_pool import ShardPool

QUEUE = 'experiments.map_sft_baselines.src.queue'


def final_config(base, policy, records):
    counts = Counter(row['cohort'] for row in records)
    if len(records) != policy['tasks'] or dict(counts) != policy['cohort_counts']:
        raise ValueError('Final evaluation cohort differs from the accepted study')
    if len({row['trajectory_id'] for row in records}) != len(records):
        raise ValueError('Final evaluation has duplicate queries')
    config = copy.deepcopy(base)
    config['data']['tasks'] = base['generalization']['task_pool']
    config['evaluation'].update(tasks=policy['tasks'], modes=policy['modes'], probe_tasks=0)
    config['execution']['evaluation_gpus'] = [policy['gpu']]
    return config


def completed(root, baseline, step, config):
    path = root / 'evaluation' / baseline / f'checkpoint-{step}' / 'summary.json'
    if path.exists():
        row = json.loads(path.read_text())
    else:
        curve = root/'results'/baseline/'curve.json'
        rows = json.loads(curve.read_text()) if curve.exists() else []
        row = next((item for item in rows if item.get('step') == step), {})
    return row.get('step') == step and all(
        row.get('metrics', {}).get(mode, {}).get('tasks') == config['evaluation']['tasks']
        for mode in config['evaluation']['modes'])


def stop_dispatcher(pid, root):
    """Stop only a verified dispatcher, leaving its independent shard workers."""
    path = Path(f'/proc/{pid}/cmdline')
    if not path.exists() or not path.read_bytes():
        return
    argv = path.read_bytes().decode().split('\0')
    if QUEUE not in argv or '--root' not in argv or argv[argv.index('--root') + 1] != str(root):
        raise RuntimeError('PID is not this study evaluation dispatcher')
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not path.exists() or not path.read_bytes():
            return
        time.sleep(.1)
    raise RuntimeError('Evaluation dispatcher did not exit')


def spawn(argv, code, log, credentials=None):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false')
    with log.open('a') as stream:
        process = subprocess.Popen(argv, cwd=code, env=env,
            stdin=subprocess.PIPE if credentials is not None else subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    if credentials is not None:
        process.stdin.write(json.dumps(credentials).encode())
        process.stdin.close()
    return process


def drain_workers(root, base, pool, policy, status):
    """Finish/publish any outer shard, including one between Popen and snapshot."""
    state = json.loads((root / 'evaluation_queue.json').read_text())
    stop_dispatcher(state['pid'], root)
    jobs = {str(job[-1]): (pid, job) for _, pid, job in
            existing_jobs(SimpleNamespace(root=root), base)}
    for item in state.get('active', {}).values():
        job = list(item['job']); job[-2:] = map(Path, job[-2:])
        jobs.setdefault(str(job[-1]), (item['pid'], tuple(job)))
    preparing = state.get('preparing')
    if preparing and str(preparing['job'][-1]) not in jobs:
        job = list(preparing['job']); job[-2:] = map(Path, job[-2:])
        owner = pool.owner(job)
        if owner:
            if owner['node'] != state['node']:
                raise RuntimeError('Prepared claim belongs to another node')
            pool.release(job, owner)
    active = []
    for pid, job in jobs.values():
        owner = pool.owner(job)
        if owner is None:
            if pool.exists(pool.output(job) / 'summary.json'):
                continue
            raise RuntimeError('ID worker is missing its claim')
        if owner['node'] != state['node']:
            raise RuntimeError('ID worker claim belongs to another node')
        proc = Path(f'/proc/{pid}/cmdline')
        worker = ExistingWorker(pid, job[-1]) if proc.exists() and proc.read_bytes() else None
        active.append((worker, job, owner))
    while active:
        status('draining_id_shard', remaining=len(active))
        for item in list(active):
            worker, job, owner = item
            if worker is not None and worker.poll() is None:
                continue
            if not (Path(job[-1]) / 'summary.json').exists():
                raise RuntimeError('ID worker exited without a complete shard; claim retained')
            pool.publish(job, owner)
            active.remove(item)
        if active:
            time.sleep(policy['drain_poll_seconds'])
    atomic_json(root / 'evaluation_queue.json', dict(state, status='paused_for_final_evaluation',
        active={}, preparing=None, updated=time.time()))


def drain_id(root, base, profile, credentials, policy, status):
    from experiments.llm_task_sft.src.bridge import connect
    jump, target = connect(profile, credentials)
    files = target.open_sftp()
    files.get_channel().settimeout(60)
    try:
        drain_workers(root, base, ShardPool(profile['target_run'], files), policy, status)
    finally:
        files.close(); target.close(); jump.close()


def remote(target, script, **values):
    from experiments.llm_task_sft.src.bridge import execute
    # All values are public runtime paths/configuration. Credentials never enter commands.
    python = values.pop('_python', 'python3')
    prefix = 'import json; v=json.loads(' + repr(json.dumps(values)) + ')\n'
    return execute(target, shlex.quote(python) + ' -c ' + shlex.quote(prefix + script))


def stop_owned(pid, expected):
    """Stop a controller/bridge only after checking its process identity."""
    path = Path(f'/proc/{pid}/cmdline')
    if not path.exists() or not path.read_bytes():
        return
    if expected not in path.read_bytes().decode().split('\0'):
        raise RuntimeError('Controller PID identity changed')
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not path.exists() or not path.read_bytes():
            return
        time.sleep(.1)
    raise RuntimeError('Controller did not exit')


def cache_evictions(entries, incoming_bytes, maximum_bytes, pending, archived):
    """Plan only recoverable, idle ID-cache evictions; final endpoints stay pinned."""
    total = sum(item['size'] for item in entries) + incoming_bytes
    selected = []
    for item in sorted(entries, key=lambda x: x['size'], reverse=True):
        key = (item['baseline'], item['step'])
        if total <= maximum_bytes:
            break
        if key in pending or key not in archived:
            continue
        selected.append(item)
        total -= item['size']
    if total > maximum_bytes:
        raise RuntimeError('Final weights cannot fit the existing cache budget')
    return selected


def run(args, credentials):
    from experiments.llm_task_sft.src.bridge import connect
    from experiments.map_sft_baselines.src.bridge import publish, return_scores
    root = args.root.resolve()
    policy = json.loads(args.policy.read_text())
    base = json.loads((root / 'config.json').read_text())
    final = root / policy['output_subdirectory']
    data = args.data.resolve() / policy['data_subdirectory']
    config = final_config(base, policy, json.loads((data / 'evaluation.json').read_text()))
    final.mkdir(exist_ok=True)
    atomic_json(final / 'config.json', config)
    profile = json.loads((root / 'bridge_profile.json').read_text())
    script_code = Path(__file__).resolve().parents[3]
    remote_id = Path(profile['target_run'])
    remote_base = remote_id.parents[1]
    remote_final = remote_id / policy['output_subdirectory']
    remote_code = remote_base / 'code' / script_code.name
    remote_python = str(remote_base / '.venv/bin/python')
    final_profile = dict(profile, target_run=str(remote_final), run_path=final,
        baselines=list(base['baselines']), early_stop=dict(base['early_stop'], mode='flag_only'))
    # Persist a public profile for the existing 028 cross-machine evaluator.
    atomic_json(final/'coordination_profile.json',
                {k:v for k,v in final_profile.items() if k != 'run_path'})
    controller = final / 'controller.json'
    def status(phase, **updates):
        atomic_json(controller, dict(pid=os.getpid(), phase=phase, updated=time.time(), **updates))
    with FileLock(str(final / 'controller.lock'), timeout=0):
        jump, target = connect(profile, credentials)
        target.get_transport().set_keepalive(30)
        files = target.open_sftp()
        files.get_channel().settimeout(120)
        saved = final/'id_handoff.json'
        if saved.exists():
            handoff = json.loads(saved.read_text())
        else:
            # Upgrade the previous 028-only handoff, whose ID dispatcher is paused.
            old_code = root.parents[1]/'code'/root.name
            local_argv = [sys.executable,'-u','-m',QUEUE,'evaluate','--config',str(root/'config.json'),
                '--root',str(root),'--data',str(args.data.resolve()),'--model-path',str(args.model_path),
                '--q-checkpoint',str(args.data.resolve()/'blocks/qmap.pt'),'--node-id','028-map-encoder-id',
                '--gpus',str(policy['gpu']),'--coordination-profile',str(root/'bridge_profile.json'),
                '--resume-workers']
            remote_launch = json.loads(remote(target, """
from pathlib import Path
r=Path(v['root']);s=json.loads((r/'evaluation_queue.json').read_text());p=Path('/proc')/str(s['pid'])
print(json.dumps(dict(argv=[a for a in (p/'cmdline').read_bytes().decode().split('\\0') if a],code=str((p/'cwd').resolve()))))
""", root=str(remote_id)))
            if '--resume-workers' not in remote_launch['argv']:
                remote_launch['argv'].append('--resume-workers')
            runtime = json.loads((root/'evaluation_runtime.json').read_text())
            bridge_pid = runtime['bridge_pid']
            bridge_proc = Path('/proc')/str(bridge_pid)
            handoff = dict(local=dict(argv=local_argv,code=str(old_code)), remote=remote_launch,
                bridge=dict(argv=[a for a in (bridge_proc/'cmdline').read_bytes().decode().split('\0') if a],
                            code=str((bridge_proc/'cwd').resolve())), bridge_pid=bridge_pid, paused=False)
            atomic_json(saved,handoff)

        def remote_spawn(argv, code, log):
            return int(remote(target, """
import os,subprocess
from pathlib import Path
p=Path(v['log']);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('a') as stream:
 child=subprocess.Popen(v['argv'],cwd=v['code'],env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(child.pid)
""", argv=argv,code=str(code),log=str(log)).strip())

        def remote_drain(where, config, phase):
            status(phase)
            remote(target, """
import sys
from pathlib import Path
sys.path.insert(0,v['code'])
from experiments.map_encoder_sft.src.final_evaluation import drain_workers
from experiments.map_sft_baselines.src.shard_pool import ShardPool
from grounded_llm.core.files import atomic_json
r=Path(v['root']);policy=v['policy']
if (r/'evaluation_queue.json').exists():
 drain_workers(r,v['config'],ShardPool(r),policy,lambda phase,**kw:atomic_json(r/'handoff_status.json',dict(phase=phase,**kw)))
""",_python=remote_python,code=str(remote_code),root=str(where),config=config,policy=policy)

        def pause_id():
            if handoff['paused']:
                return
            status('draining_id_for_final')
            bridge_proc=Path('/proc')/str(handoff['bridge_pid'])/'cmdline'
            while bridge_proc.exists() and bridge_proc.read_bytes():
                bridge_state=json.loads((root/'bridge_status.json').read_text())
                if bridge_state.get('phase') in ('idle','connect'):
                    break
                status('waiting_for_id_weight_transfer')
                time.sleep(policy['drain_poll_seconds'])
            stop_owned(handoff['bridge_pid'],'experiments.map_sft_baselines.src.bridge')
            drain_id(root,base,profile,credentials,policy,status)
            remote_drain(remote_id,base,'draining_4090_id_for_final')
            handoff['paused']=True;atomic_json(saved,handoff)

        def stage(pending):
            status('staging_final',endpoints=pending)
            already_done = {b:s for b,s in endpoints.items() if b not in pending}
            remote(target,"""
from pathlib import Path
for b,s in v['done'].items():
 p=Path(v['root'])/'models'/b/('checkpoint-'+str(s))
 (p/'weights.pt').unlink(missing_ok=True)
""",root=str(remote_final),done=already_done)
            # Hard links reuse already-cached bytes. Only recoverable idle ID
            # cache entries can be evicted to keep the existing total budget.
            entries=json.loads(remote(target,"""
from pathlib import Path
r=Path(v['root']);out=[]
for p in (r/'models').glob('B*/checkpoint-*/weights.pt'):
 marker=p.with_name('evaluation_ready.json')
 out.append(dict(baseline=p.parents[1].name,step=int(p.parent.name[11:]),size=p.stat().st_size,meta=json.loads(marker.read_text()) if marker.exists() else None))
print(json.dumps(out))
""",root=str(remote_id)))
            archived=set()
            for item in entries:
                ck=root/'training'/item['baseline']/f"checkpoint-{item['step']}"
                if ((ck/'weights.pt').is_file() and (ck/'evaluation_ready.json').is_file()
                    and (ck/'weights.pt').stat().st_size == item['size']
                    and json.loads((ck/'evaluation_ready.json').read_text()) == item['meta']):
                    archived.add((item['baseline'],item['step']))
            cached={(i['baseline'],i['step']) for i in entries}
            incoming=sum((root/'training'/b/f'checkpoint-{s}'/'weights.pt').stat().st_size
                         for b,s in pending.items() if (b,s) not in cached)
            budget=json.loads((script_code/'experiments/map_encoder_sft/configs/evaluation_cache.json').read_text())['maximum_weight_bytes']
            evict=cache_evictions(entries,incoming,budget,set(pending.items()),archived)
            remote(target,"""
from pathlib import Path
r=Path(v['root'])
for item in v['evict']:
 p=r/'models'/item['baseline']/('checkpoint-'+str(item['step']))
 for name in ['evaluation_ready.json','weights.pt']:
  (p/name).unlink(missing_ok=True)
Path(v['final']).mkdir(parents=True,exist_ok=True)
""",root=str(remote_id),final=str(remote_final),evict=evict)
            publish(files,final/'config.json',str(remote_final/'config.json'))
            for baseline,step in pending.items():
                source=root/'training'/baseline/f'checkpoint-{step}'
                dest=final/'models'/baseline/source.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.symlink_to(source)
                rd=remote_final/'models'/baseline/source.name
                linked=remote(target,"""
from pathlib import Path
import os
p=Path(v['dest']);p.mkdir(parents=True,exist_ok=True);src=Path(v['id'])/'models'/v['baseline']/p.name/'weights.pt'
if not (p/'weights.pt').exists() and src.exists():os.link(src,p/'weights.pt')
(p/'weights_retired.json').unlink(missing_ok=True)
print((p/'weights.pt').exists())
""",dest=str(rd),id=str(remote_id),baseline=baseline).strip()=='True'
                if not linked:publish(files,source/'weights.pt',str(rd/'weights.pt'))
                for name in ('state.json','evaluation_ready.json'):
                    publish(files,source/name,str(rd/name))
            # Preserve every already-completed 028-only final shard on upgrade.
            for summary in final.glob('evaluation/B*/checkpoint-*/*/summary.json'):
                output=remote_final/summary.parent.relative_to(final)
                try:files.stat(str(output/'summary.json'));continue
                except FileNotFoundError:pass
                remote(target,"from pathlib import Path;Path(v['path']).mkdir(parents=True,exist_ok=True)",path=str(output))
                publish(files,summary.with_name('cases.jsonl'),str(output/'cases.jsonl'))
                publish(files,summary,str(output/'summary.json'))

        def resume_id():
            local=spawn(handoff['local']['argv'],handoff['local']['code'],root/'logs/evaluate_028_after_final.log',credentials)
            rp=remote_spawn(handoff['remote']['argv'],handoff['remote']['code'],remote_id/'evaluation_after_final.log')
            bridge=spawn(handoff['bridge']['argv'],handoff['bridge']['code'],root/'logs/bridge_after_final.log',credentials)
            runtime=json.loads((root/'evaluation_runtime.json').read_text())
            atomic_json(root/'evaluation_runtime.json',dict(runtime,local_pid=local.pid,remote_pid=rp,bridge_pid=bridge.pid))
            handoff.update(paused=False,bridge_pid=bridge.pid);atomic_json(saved,handoff)

        evaluators=[]
        try:
            # A previous 028-only final dispatcher may still own an unfinished shard.
            if (final/'evaluation_queue.json').exists():
                status('draining_previous_final')
                state=json.loads((final/'evaluation_queue.json').read_text())
                proc=Path('/proc')/str(state['pid'])/'cmdline'
                argv=proc.read_bytes().decode().split('\0') if proc.exists() else []
                if '--coordination-profile' in argv or handoff.get('final_shared_pool'):
                    drain_id(final,config,final_profile,credentials,policy,status)
                else:
                    drain_workers(final,config,ShardPool(final),policy,status)
            remote_drain(remote_final,config,'draining_previous_4090_final')
            while True:
                endpoints = {}
                for baseline in base['baselines']:
                    path = root / 'training' / baseline / 'finished.json'
                    if not path.exists():
                        continue
                    finished = json.loads(path.read_text())
                    if finished['status'] not in ('completed', 'stopped'):
                        raise RuntimeError('Training did not produce an accepted endpoint')
                    step = finished['step']
                    ck = path.parent / f'checkpoint-{step}'
                    if not all((ck / n).is_file() for n in ('weights.pt', 'evaluation_ready.json', 'state.json')):
                        raise RuntimeError('Finished endpoint lacks complete inference files')
                    endpoints[baseline] = step
                pending = {b:s for b,s in endpoints.items() if not completed(final,b,s,config)}
                if not pending:
                    if handoff['paused']:
                        resume_id()
                    if len(endpoints) == len(base['baselines']):
                        status('completed', endpoints=endpoints)
                        return
                    status('waiting_for_training', completed_endpoints=endpoints)
                    time.sleep(policy['poll_seconds'])
                    continue
                pause_id()
                stage(pending)
                handoff['final_shared_pool']=True;atomic_json(saved,handoff)
                argv = [sys.executable, '-u', '-m', QUEUE, 'evaluate', '--config',
                    str(final / 'config.json'), '--root', str(final), '--data', str(data),
                    '--model-path', str(args.model_path), '--q-checkpoint', str(data/'blocks/qmap.pt'),
                    '--node-id', '028-map-encoder-final', '--gpus', str(policy['gpu']),
                    '--coordination-profile',str(final/'coordination_profile.json'),'--resume-workers']
                evaluator = spawn(argv, script_code, final / 'evaluation.log',credentials)
                remote_argv=[remote_python,'-u','-m',QUEUE,'evaluate','--config',str(remote_final/'config.json'),
                    '--root',str(remote_final),'--data',str(remote_base/'data'/root.name/policy['data_subdirectory']),
                    '--model-path',str(remote_base/'models/Qwen2.5-1.5B-Instruct'),
                    '--q-checkpoint',str(remote_base/'data'/root.name/policy['data_subdirectory']/'blocks/qmap.pt'),
                    '--node-id','4090-map-encoder-final','--gpus',','.join(map(str,policy['remote_gpus'])),'--resume-workers']
                rp=remote_spawn(remote_argv,remote_code,remote_final/'evaluation.log')
                evaluators=[evaluator.pid,rp]
                while not all(completed(final,b,s,config) for b,s in pending.items()):
                    if evaluator.poll() is not None:
                        raise RuntimeError('Final evaluation dispatcher exited; inspect retained claims')
                    return_scores(files,final_profile,config)
                    remote_state=json.loads(remote(target,"""
from pathlib import Path
p=Path(v['path']);proc=Path('/proc')/str(v['pid'])/'cmdline'
if not proc.exists() or not proc.read_bytes():raise RuntimeError('Remote final dispatcher exited')
print(p.read_text() if p.exists() else '{}')
""",path=str(remote_final/'evaluation_queue.json'),pid=rp))
                    if remote_state.get('failed'):
                        raise RuntimeError('4090 final queue reports a failed shard')
                    status('evaluating_final', endpoints=pending, evaluator_pid=evaluator.pid,remote_evaluator_pid=rp,
                           priority=policy['priority'])
                    time.sleep(policy['poll_seconds'])
                # Completed summaries cover all modes; no unfinished shard remains in this batch.
                state = json.loads((final / 'evaluation_queue.json').read_text())
                while state.get('active') or state.get('preparing'):
                    time.sleep(policy['poll_seconds'])
                    state = json.loads((final / 'evaluation_queue.json').read_text())
                stop_dispatcher(evaluator.pid, final)
                remote_drain(remote_final,config,'finishing_final_batch')
                # Scores are retained; disposable links no longer pin cache bytes.
                remote(target,"""
from pathlib import Path
r=Path(v['root'])
for b,s in v['endpoints'].items():
 p=r/'models'/b/('checkpoint-'+str(s));(p/'weights.pt').unlink(missing_ok=True)
 (p/'weights_retired.json').write_text(json.dumps(dict(reason='final_evaluation_complete',baseline=b,step=s)))
""",root=str(remote_final),endpoints=pending)
                evaluators=[]
        except Exception as error:
            status('failed', error=str(error),evaluators=evaluators)
            raise
        finally:
            files.close();target.close();jump.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('root', 'data', 'model-path', 'policy'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args(), json.load(sys.stdin))


if __name__ == '__main__':
    main()
