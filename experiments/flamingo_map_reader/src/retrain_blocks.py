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

from filelock import FileLock

from .trajectory_queue import clear_partial_shard, resume_point


def main():
    parser = ArgumentParser()
    for name in ('config', 'source-manifest', 'model-path', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prepared-manifest', type=Path,
                        help='Reuse identical prepared data after interface, batch or checkpoint changes')
    parser.add_argument('--resume', action='store_true',
                        help='Continue the existing output from its newest complete checkpoint')
    parser.add_argument('--resume-topology', type=Path,
                        help='Explicit equivalent-global-batch migration contract; requires --resume')
    parser.add_argument('--training-only', action='store_true',
                        help='Leave final evaluation to the separate checkpoint queue')
    args = parser.parse_args()
    if args.out.exists() and not args.resume:
        raise FileExistsError(args.out)
    if args.resume and not (args.out/'data/manifest.json').is_file():
        raise ValueError('Resume requires an existing prepared manifest in the same output')
    if args.resume_topology and not args.resume:
        raise ValueError('Topology migration requires --resume')
    if args.resume_topology and not args.training_only:
        raise ValueError('Two-GPU continuation needs --training-only and a separate final evaluation queue')
    args.out.mkdir(parents=True, exist_ok=args.resume)
    with FileLock(str(args.out/'.pipeline.lock'), timeout=0):
        run(args)


def run(args):
    """Use the same training and evaluation modules for fresh and resumed runs."""
    logs = args.out/'logs'
    logs.mkdir(exist_ok=args.resume)
    config = json.loads(args.config.read_text())
    if config['task'] != 'blocks' or config['training']['world_size'] != 4:
        raise ValueError('This run requires blocks and four GPUs')
    topology = None
    if args.resume_topology:
        from .train import validate_resume_topology
        topology = json.loads(args.resume_topology.read_text())
        validate_resume_topology(config, topology, topology['world_size'], topology['per_device_batch_size'])
    world_size = topology['world_size'] if topology else config['training']['world_size']
    state = dict(status='running', phase='prepare', run=str(args.out.resolve()))

    def status(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = args.out/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(args.out/'status.json')
        print(json.dumps(state), flush=True)

    def execute(name, command, env=None):
        with (logs/(name+'.log')).open('a' if args.resume else 'w') as log:
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
                f'--nproc_per_node={world_size}', '-m', module+name]

    try:
        status()
        if args.resume:
            if json.loads((args.out/'data/manifest.json').read_text())['config'] != config:
                raise ValueError('Resume must use the existing run configuration')
        elif args.prepared_manifest:
            prepared = json.loads(args.prepared_manifest.read_text())
            before = {k:v for k,v in prepared['config'].items() if k not in ('map', 'checkpoint')}
            after = {k:v for k,v in config.items() if k not in ('map', 'checkpoint')}
            # Failure histories have cached copies too; retaining their first
            # copy changes sampling only, without rebuilding labels or maps.
            failure_variants = config['data'].get('failure_numbering_variants')
            for value in (before, after):
                value['data'] = {k:v for k,v in value['data'].items()
                                 if k != 'failure_numbering_variants'}
            for value in (before, after):
                value['training'] = {k:v for k,v in value['training'].items()
                    if k not in ('checkpoint_layers', 'loss_chunk_tokens',
                                 'batch_size_candidates_per_device', 'batch_size_candidates_global')}
            if before != after:
                raise ValueError('Prepared data reuse requires identical labels, sampling and training budget')
            prepared['config'] = config
            prepared['prepared_data_source'] = str(args.prepared_manifest.resolve())
            if failure_variants is not None:
                for record in prepared['records']:
                    if record['split'] == 'train' and record.get('no_solution'):
                        if not 1 <= failure_variants <= record['variants']:
                            raise ValueError('Requested failure numbering copies are not cached')
                        record['variants'] = failure_variants
                prepared['audit']['training_numbered_trajectories'] = sum(
                    r['variants'] for r in prepared['records'] if r['split'] == 'train')
                prepared['audit']['failure_numbering_variants'] = failure_variants
            (args.out/'data').mkdir()
            (args.out/'data/trajectories').symlink_to(
                (args.prepared_manifest.parent/'trajectories').resolve(), target_is_directory=True)
            (args.out/'data/manifest.json').write_text(json.dumps(prepared, indent=2)+'\n')
            (args.out/'data/audit.json').write_text(json.dumps(prepared['audit'], indent=2)+'\n')
        else:
            execute('prepare', python('prepare_blocks_retrain') + [
                '--config', str(args.config), '--source-manifest', str(args.source_manifest),
                '--model-path', str(args.model_path), '--out', str(args.out/'data'),
                '--workers', str(args.workers)])
        manifest_path = args.out/'data/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        status(phase='probe', audit=manifest['audit'])
        batch = None
        final = args.out/'training/models/final'
        training_done = (final/'evaluation_ready.json').is_file() and (final/'adapter.pt').is_file()
        resume = None
        if args.resume and not training_done:
            resume = resume_point(args.out/'training/models', world_size)
            if resume is None:
                raise ValueError('No complete training checkpoint exists; refusing to restart silently')
            saved = json.loads((args.out/'training/config.json').read_text())
            batch = topology['per_device_batch_size'] if topology else saved['batch_size']
        for candidate in (() if args.resume else config['training']['batch_size_candidates_per_device']):
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
        if batch is None and not training_done:
            raise RuntimeError('No requested batch fits; all probes failed with OOM')
        if not training_done:
            status(phase='training', per_device_batch=batch, global_batch=world_size*batch,
                   world_size=world_size, resume_topology=topology,
                   resume_checkpoint=str(resume) if resume else None)
            execute('training', distributed('train') + ['--config', str(args.config),
                '--manifest', str(manifest_path), '--model-path', str(args.model_path),
                '--q-checkpoint', manifest['q_checkpoint'], '--batch-size', str(batch),
                '--out', str(args.out/'training')] + (['--resume', str(resume)] if resume else []) +
                (['--resume-topology', str(args.resume_topology)] if topology else []))
        if args.training_only:
            status(status='training_completed', phase='final_evaluation_on_separate_queue')
            return
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
                output = args.out/'evaluation'/name
                if args.resume and (output/'summary.json').is_file():
                    continue
                clear_partial_shard(output)
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
