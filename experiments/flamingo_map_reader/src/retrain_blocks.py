"""One authorized run: revised data, measured four-GPU batch, fresh SFT, final tests."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = ArgumentParser()
    for name in ('config', 'source-manifest', 'model-path', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    logs = args.out/'logs'
    logs.mkdir()
    config = json.loads(args.config.read_text())
    if config['task'] != 'blocks' or config['training']['world_size'] != 4:
        raise ValueError('This run requires blocks and four GPUs')
    state = dict(status='running', phase='prepare', run=str(args.out.resolve()))

    def status(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = args.out/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.out/'status.json')
        print(json.dumps(state), flush=True)

    def execute(name, command, env=None):
        with (logs/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env)
            code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)

    module = 'experiments.flamingo_map_reader.src.'
    def python(name):
        return [sys.executable, '-m', module+name]

    def distributed(name):
        return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node=4', '-m', module+name]

    try:
        status()
        execute('prepare', python('prepare_blocks_retrain') + [
            '--config', str(args.config), '--source-manifest', str(args.source_manifest),
            '--model-path', str(args.model_path), '--out', str(args.out/'data'),
            '--workers', str(args.workers)])
        manifest_path = args.out/'data/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        status(phase='probe', audit=manifest['audit'])
        batch = None
        for candidate in config['training']['batch_size_candidates_per_device']:
            name = f'probe_batch_{candidate}'
            try:
                execute(name, distributed('probe_training_batch') + [
                    '--config', str(args.config), '--manifest', str(manifest_path),
                    '--model-path', str(args.model_path), '--out', str(args.out/name),
                    '--batch-size', str(candidate)])
                batch = candidate
                break
            except subprocess.CalledProcessError:
                log = (logs/(name+'.log')).read_text()
                if 'out of memory' not in log.lower():
                    raise
        if batch is None:
            raise RuntimeError('No requested batch fits; all probes failed with OOM')
        status(phase='training', per_device_batch=batch, global_batch=4*batch)
        execute('training', distributed('train') + ['--config', str(args.config),
            '--manifest', str(manifest_path), '--model-path', str(args.model_path),
            '--q-checkpoint', manifest['q_checkpoint'], '--batch-size', str(batch),
            '--out', str(args.out/'training')])
        checkpoint = args.out/'training/models/final/adapter.pt'
        status(phase='evaluation', checkpoint=str(checkpoint))
        jobs = []
        for split, mode in (('test', 'reference'), ('test', 'rollout'), ('test_no_solution', 'reference')):
            count = sum(r['split'] == split for r in manifest['records'])
            for start in range(0, count, config['evaluation']['shard_tasks']):
                stop = min(start+config['evaluation']['shard_tasks'], count)
                jobs.append((split, mode, start, stop))
        # Each lane owns one GPU for its whole serial shard list.
        def lane(gpu):
            for split, mode, start, stop in jobs[gpu::4]:
                name = f'{split}_{mode}_{start:05d}_{stop:05d}'
                execute(name, python('trajectory_eval') + ['--manifest', str(manifest_path),
                    '--adapter-checkpoint', str(checkpoint), '--model-path', str(args.model_path),
                    '--out', str(args.out/'evaluation'/name), '--split', split, '--mode', mode,
                    '--start', str(start), '--stop', str(stop),
                    '--variants', str(config['evaluation']['numbering_variants'])],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lane, range(4)))
        from .trajectory_eval import aggregate
        summary = {}
        for split, mode in (('test', 'reference'), ('test', 'rollout'), ('test_no_solution', 'reference')):
            cases = []
            for path in sorted((args.out/'evaluation').glob(f'{split}_{mode}_*/cases.jsonl')):
                cases.extend(json.loads(line) for line in path.read_text().splitlines())
            summary[f'{split}_{mode}'] = aggregate(cases, manifest)
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        status(status='completed', phase='completed')
    except Exception as error:
        status(status='failed', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
