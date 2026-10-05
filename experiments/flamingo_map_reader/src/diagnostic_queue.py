"""Two disjoint workers: complete paired reference first, then assisted rollouts."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def jobs(run, output, protocol):
    stages = [[], []]
    for task in protocol['stages'][0]['tasks']:
        manifest = json.loads((run / task / 'data/manifest.json').read_text())
        count = sum(r['split'] == protocol['split'] for r in manifest['records'])
        for start in range(0, count, protocol['shard_tasks']):
            stop = min(count, start + protocol['shard_tasks'])
            stages[0].append(dict(task=task, kind='q_reverse_reference', helper_steps=0,
                start=start, stop=stop, out=str(output / task / 'q_reverse_reference' / f'{start:05d}_{stop:05d}')))
    manifest = json.loads((run / 'blocks/data/manifest.json').read_text())
    count = sum(r['split'] == protocol['split'] for r in manifest['records'])
    for helper in protocol['stages'][1]['helper_steps']:
        for start in range(0, count, protocol['shard_tasks']):
            stop = min(count, start + protocol['shard_tasks'])
            stages[1].append(dict(task='blocks', kind='safe_prefix_rollout', helper_steps=helper,
                start=start, stop=stop, out=str(output / 'blocks' / f'safe_prefix_{helper}_rollout' / f'{start:05d}_{stop:05d}')))
    return stages


def owned_jobs(stage, index, worker):
    return [j for i, j in enumerate(stage) if i % 2 == worker]


def read_cases(group):
    return [json.loads(line) for shard in sorted(group.glob('*/summary.json'))
            for line in (shard.parent / 'cases.jsonl').read_text().splitlines()]


def main():
    parser = ArgumentParser()
    for name in ('run-root', 'model-path', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker-index', choices=(0, 1), type=int, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    contract = args.out / 'protocol.json'
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError('Output belongs to another diagnostic protocol')
    atomic_json(args.out / f'protocol_worker_{args.worker_index}.json', protocol)
    stages = jobs(args.run_root, args.out, protocol)
    status_path = args.out / f'worker_{args.worker_index}_status.json'
    completed = []
    for index, stage in enumerate(stages):
        pending = [j for j in owned_jobs(stage, index, args.worker_index) if not (Path(j['out']) / 'summary.json').is_file()]
        running = {}
        while pending or running:
            for gpu, active in list(running.items()):
                code = active['process'].poll()
                if code is None:
                    continue
                if code or not (Path(active['job']['out']) / 'summary.json').is_file():
                    failure = dict(job=active['job'], exit_code=code, log=active['log'])
                    atomic_json(args.out / f'worker_{args.worker_index}_failed.json', failure)
                    for other in running.values():
                        if other['process'].poll() is None:
                            other['process'].terminate()
                    raise RuntimeError('Diagnostic failed: ' + json.dumps(failure))
                completed.append(active['job'])
                del running[gpu]
            for gpu in args.gpus:
                if gpu in running or not pending:
                    continue
                job = pending.pop(0)
                destination = Path(job['out'])
                if destination.exists():
                    # Preserve interrupted output; never remove previous data.
                    destination.rename(destination.with_name(destination.name + '.interrupted.' + str(time.time_ns())))
                command = [sys.executable, '-u', '-m', 'experiments.flamingo_map_reader.src.diagnostic_eval',
                    '--manifest', str(args.run_root / job['task'] / 'data/manifest.json'),
                    '--adapter-checkpoint', str(args.run_root / job['task'] / 'training/models/final/adapter.pt'),
                    '--model-path', str(args.model_path), '--protocol', str(args.protocol),
                    '--out', job['out'], '--kind', job['kind'], '--helper-steps', str(job['helper_steps']),
                    '--start', str(job['start']), '--stop', str(job['stop'])]
                log = args.out / ('_'.join(destination.relative_to(args.out).parts) + '.log')
                with log.open('wb') as handle:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false'))
                running[gpu] = dict(process=process, job=job, log=str(log))
                print(json.dumps(dict(event='launched', gpu=gpu, pid=process.pid, job=job)), flush=True)
            atomic_json(status_path, dict(pid=os.getpid(), stage=index + 1, pending=len(pending), completed=len(completed),
                running={gpu: dict(pid=a['process'].pid, job=a['job'], log=a['log']) for gpu, a in running.items()}))
            time.sleep(10)
        while not all((Path(j['out']) / 'summary.json').is_file() for j in stage):
            failures = list(args.out.glob('worker_*_failed.json'))
            if failures:
                raise RuntimeError('Another diagnostic worker failed: ' + str(failures))
            atomic_json(status_path, dict(pid=os.getpid(), stage=index + 1, waiting_for_other_worker=True,
                                         completed=len(completed), running={}))
            time.sleep(15)
        print(json.dumps(dict(event='stage_complete', stage=index + 1)), flush=True)
    if args.worker_index == 0:
        from .diagnostic_eval import paired_summary, assisted_summary
        atomic_json(contract, protocol)
        for task in ('path', 'blocks'):
            group = args.out / task / 'q_reverse_reference'
            atomic_json(group / 'summary.json', paired_summary(read_cases(group)))
        manifest = json.loads((args.run_root / 'blocks/data/manifest.json').read_text())
        for helper in protocol['stages'][1]['helper_steps']:
            group = args.out / 'blocks' / f'safe_prefix_{helper}_rollout'
            atomic_json(group / 'summary.json', assisted_summary(read_cases(group), manifest))
        baseline = args.run_root / 'evaluation/blocks/final/validation_rollout_map'
        cases = read_cases(baseline)
        expected = sum(r['split'] == protocol['split'] for r in manifest['records'])
        if len(cases) != expected or any(c['variant'] != 0 for c in cases):
            raise RuntimeError('Assisted baseline is not the complete matched validation battery')
        records = {r['trajectory_id']: r for r in manifest['records']}
        for case in cases:
            record = records[case['trajectory_id']]
            case['goal_type'] = 'initial_goal' if record['start'] == record['goal'] else 'empty' if int(record['goal']) == 0 else 'nonempty'
            for row in case['turns']:
                row['assisted'] = False
        atomic_json(args.out / 'blocks/baseline_rollout_summary.json', assisted_summary(cases, manifest))
    atomic_json(args.out / f'worker_{args.worker_index}_completion.json', dict(success=True, completed=len(completed)))
    atomic_json(status_path, dict(pid=os.getpid(), stage='complete', running={}, success=True, completed=len(completed)))


if __name__ == '__main__':
    main()
