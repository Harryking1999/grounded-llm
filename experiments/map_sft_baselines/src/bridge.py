"""Return scores first, publish bounded weights, and reconnect after I/O errors."""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from filelock import FileLock
import paramiko
from experiments.llm_task_sft.src.bridge import connect, execute
from grounded_llm.core.files import atomic_json
from .contract import plateau_status, checkpoint_priority
from .retention import prune_completed, has_capacity, prioritize_pending


def publish(files, source, destination):
    temporary = destination + '.part'
    try:
        files.put(str(source), temporary)
        files.posix_rename(temporary, destination)
    except Exception:
        try:
            files.remove(temporary)
        except OSError:
            pass
        raise


def return_scores(files, profile, config):
    root = profile['run_path']
    returned = []
    for base in profile['baselines']:
        remote = profile['target_run'] + '/results/' + base
        try:
            with files.open(remote + '/curve.json') as handle:
                incoming = json.load(handle)
        except FileNotFoundError:
            continue
        path = root / 'results' / base / 'curve.json'
        # Rebuilding an evaluator must not erase previously returned checkpoints.
        existing = json.loads(path.read_text()) if path.exists() else []
        merged = {row['step']: row for row in existing}
        merged.update({row['step']: row for row in incoming})
        curve = [merged[step] for step in sorted(merged)]
        atomic_json(path, curve)
        if len(curve) == len(incoming):
            try:
                files.get(remote + '/report.md', str(path.with_name('report.md.part')))
                path.with_name('report.md.part').replace(path.with_name('report.md'))
            except FileNotFoundError:
                pass
        flag = plateau_status(curve, profile['early_stop'], config['training']['save_steps'])
        if flag:
            atomic_json(path.with_name('plateau.json'), flag)
            if profile['early_stop']['mode'] == 'automatic':
                (root / 'training' / base / 'stop_after_checkpoint').touch()
        else:
            path.with_name('plateau.json').unlink(missing_ok=True)
        returned.append(base)
    return returned


def record_progress(profile, progress, **updates):
    progress.update(updates, updated=time.time())
    atomic_json(profile['run_path'] / 'bridge_status.json', progress)


def cycle(files, target, profile, config, policy, progress):
    record_progress(profile, progress, status='running', phase='results')
    progress['returned'] = return_scores(files, profile, config)
    progress['scores_synced_at'] = time.time()
    ready = sorted((profile['run_path'] / 'training').glob('B*/checkpoint-*/evaluation_ready.json'),
                   key=lambda p: checkpoint_priority(json.loads(p.read_text())['step'], config))
    archived = {(p.parent.parent.name, p.parent.name): p.parent / 'weights.pt' for p in ready}
    record_progress(profile, progress, phase='retention')
    progress['deleted'] = prune_completed(files, profile, config, archived)
    if profile.get('source_retention') == 'latest_and_best':
        progress['source_deleted'] = prune_source(profile, config, ready)
    if config.get('evaluation', {}).get('priority_step_interval'):
        preferred, deferred = prioritize_pending(files, profile, config, ready, archived)
        progress['preferred_steps'] = preferred
        progress['deferred_cache'] = deferred
        ready = [p for p in ready if json.loads(p.read_text())['step'] ==
                 preferred.get(p.parent.parent.name)]
    record_progress(profile, progress, phase='weights')
    published, blocked, seen = [], [], set()
    for marker in ready:
        base = marker.parent.parent.name
        if (base not in profile.get('publish_baselines', profile['baselines']) or
                base in config['execution'].get('retired_baselines', [])):
            continue
        if policy['latest_only'] and base in seen:
            continue
        seen.add(base)
        dest = profile['target_run'] + '/models/' + base + '/' + marker.parent.name
        try:
            files.stat(dest + '/evaluation_ready.json')
            continue
        except FileNotFoundError:
            pass
        weights = marker.parent / 'weights.pt'
        if not has_capacity(files, profile, base, weights.stat().st_size, policy):
            blocked.append(dict(baseline=base, checkpoint=marker.parent.name))
            continue
        execute(target, 'mkdir -p ' + shlex.quote(dest))
        record_progress(profile, progress, checkpoint=str(marker.parent), phase='weights')
        for name in ('weights.pt', 'state.json', 'evaluation_ready.json'):
            publish(files, marker.parent / name, dest + '/' + name)
        published.append(dict(baseline=base, checkpoint=marker.parent.name))
        if len(published) >= policy['maximum_transfers_per_cycle']:
            break
    progress.update(published=published, blocked=blocked, phase='idle',
                    status='waiting_for_capacity' if blocked and not published else 'running')


def prune_source(profile, config, ready):
    """After full scores return, retain latest and best inference archives.

    Pending/unscored weights are never removed. The trainer independently retains
    the latest complete optimizer state. Metadata remains for run identity.
    """
    deleted=[]
    for base in profile.get('publish_baselines', profile['baselines']):
        markers=[p for p in ready if p.parent.parent.name==base]
        if not markers:
            continue
        curve_path=profile['run_path']/'results'/base/'curve.json'
        if not curve_path.exists():
            continue
        curve=json.loads(curve_path.read_text())
        from .contract import evaluation_modes,cohort_size
        complete=[row for row in curve if all(row.get('metrics',{}).get(m,{}).get('tasks')==cohort_size(config,m)
                                             for m in evaluation_modes(config,base))]
        if not complete:
            continue
        newest=max(json.loads(p.read_text())['step'] for p in markers)
        best=max(complete,key=lambda row:(row['metrics']['RO']['arrival'],row['step']))['step']
        scored={row['step'] for row in complete}
        for marker in markers:
            step=json.loads(marker.read_text())['step']
            weight=marker.parent/'weights.pt'
            if step in scored and step not in (newest,best) and weight.exists():
                weight.unlink()
                deleted.append(dict(baseline=base,step=step))
    return deleted


def run(profile_path, credentials, cycles=None):
    root = Path(json.loads(profile_path.read_text())['run'])
    root.mkdir(parents=True, exist_ok=True)
    failures, count = 0, 0
    with FileLock(str(root / 'bridge.lock'), timeout=0):
        while cycles is None or count < cycles:
            count += 1
            profile = json.loads(profile_path.read_text())
            profile['run_path'] = root
            config = json.loads(Path(profile['config']).read_text())
            policy_path = Path(profile.get('retention_config', Path(__file__).parents[1] / 'configs/bridge_retention.json'))
            policy = json.loads(policy_path.read_text())
            interval = profile.get('interval_seconds', 60)
            jump = target = files = None
            progress = dict(status='connecting', phase='connect', updated=time.time())
            atomic_json(root / 'bridge_status.json', progress)
            try:
                jump, target = connect(profile, credentials)
                target.get_transport().set_keepalive(30)
                files = paramiko.SFTPClient.from_transport(target.get_transport(), window_size=64*1024*1024, max_packet_size=1024*1024)
                files.get_channel().settimeout(120)
                cycle(files, target, profile, config, policy, progress)
                failures = 0
            except (OSError, EOFError, paramiko.SSHException, RuntimeError) as error:
                failures += 1
                interval = min(interval * 2 ** min(failures - 1, 6), policy['retry_maximum_seconds'])
                progress.update(status='retrying', error_type=type(error).__name__, error=str(error),
                                consecutive_failures=failures, retry_at=time.time() + interval)
                with (root / 'bridge_failures.jsonl').open('a') as log:
                    log.write(json.dumps(dict(progress, failed_at=time.time())) + '\n')
            finally:
                for client in (files, target, jump):
                    if client:
                        try:
                            client.close()
                        except OSError:
                            pass
                progress['updated'] = time.time()
                atomic_json(root / 'bridge_status.json', progress)
            if cycles is None or count < cycles:
                time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    run(args.profile, json.load(sys.stdin))


if __name__ == '__main__':
    main()
