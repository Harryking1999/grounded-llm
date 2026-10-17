"""Bound disposable evaluator weights while retaining metadata and scores."""
import json
from contextlib import contextmanager
from pathlib import PurePosixPath
from .contract import evaluation_modes, cohort_size
from .shard_pool import ShardPool


@contextmanager
def dispatch_lock(pool):
    path = pool.root / 'evaluation_dispatch.lock'
    try:
        pool.files.mkdir(str(path))
    except OSError:
        if pool.exists(path):
            yield False
            return
        raise
    try:
        yield True
    finally:
        pool.files.rmdir(str(path))


def cache_entries(pool):
    models = pool.root / 'models'
    if not pool.exists(models):
        return []
    entries = []
    for base in pool.files.listdir(str(models)):
        if not base.startswith('B') or PurePosixPath(base).name != base:
            continue
        for name in pool.files.listdir(str(models / base)):
            if not name.startswith('checkpoint-') or not name[11:].isdigit():
                continue
            path = models / base / name
            size = sum(item.st_size for item in pool.files.listdir_attr(str(path))
                       if item.filename in ('weights.pt', 'weights.pt.part'))
            entries.append((base, name, path, size))
    return entries


def prune_completed(files, profile, config, archived):
    """Require a source archive and mirrored scores; never evict claimed work.

    The dispatch lock prevents claims between checking and retirement. The
    tombstone prevents redispatch and keeps the bridge from republishing weights.
    """
    pool = ShardPool(profile['target_run'], files)
    deleted = []
    with dispatch_lock(pool) as locked:
        if not locked:
            return deleted
        for base, name, path, size in cache_entries(pool):
            source = archived.get((base, name))
            if not source or not source.is_file() or source.stat().st_size <= 0:
                continue
            if not pool.exists(path / 'evaluation_ready.json'):
                # An interrupted upload cannot have any legitimate claimant.
                # Discard its payload before retrying so it cannot consume a slot.
                for filename in ('weights.pt', 'weights.pt.part'):
                    item = path / filename
                    if pool.exists(item):
                        files.remove(str(item))
                if size:
                    deleted.append(dict(baseline=base, checkpoint=name, bytes=size, reason='incomplete_upload'))
                continue
            meta = pool.read(path / 'evaluation_ready.json')
            source_meta = json.loads(source.with_name('evaluation_ready.json').read_text())
            if meta != source_meta or meta['baseline'] != base or meta['step'] != int(name[11:]):
                continue
            evaluation = pool.root / 'evaluation' / base / name
            if pool.exists(evaluation) and any(x.endswith('.worker.claim') for x in files.listdir(str(evaluation))):
                continue
            retired = base in config['execution'].get('retired_baselines', [])
            completed = False
            summary = evaluation / 'summary.json'
            if not retired and base in config['baselines'] and pool.exists(summary):
                row = pool.read(summary)
                complete = (row.get('step') == meta['step'] and
                            all(row.get('metrics', {}).get(m, {}).get('tasks') == cohort_size(config, m)
                                for m in evaluation_modes(config, base)))
                curve = profile['run_path'] / 'results' / base / 'curve.json'
                completed = complete and curve.exists() and any(r == row for r in json.loads(curve.read_text()))
            tombstone = path / 'weights_retired.json'
            if not (retired or completed or pool.exists(tombstone)):
                continue
            pool.write(tombstone, dict(baseline=base, step=meta['step'],
                                      reason='retired_baseline' if retired else 'evaluation_complete',
                                      archive=str(source)))
            for filename in ('weights.pt', 'weights.pt.part'):
                item = path / filename
                if pool.exists(item):
                    files.remove(str(item))
            if size:
                deleted.append(dict(baseline=base, checkpoint=name, bytes=size))
    return deleted


def has_capacity(files, profile, baseline, size, policy):
    entries = cache_entries(ShardPool(profile['target_run'], files))
    return (sum(e[3] for e in entries) + size <= policy['maximum_weight_bytes'] and
            sum(e[0] == baseline and e[3] > 0 for e in entries) < policy['maximum_pending_checkpoints_per_baseline'])
