"""Measure the four-GPU memory boundary and a batch retaining 10% allocator headroom."""

from argparse import ArgumentParser
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = ArgumentParser()
    for key in ('config', 'manifest', 'model-path', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    module = 'experiments.flamingo_map_reader.src.'
    profile = args.out/'shapes.json'
    with (args.out/'profile.log').open('w') as log:
        subprocess.run([sys.executable, '-m', module+'profile_training_shapes',
            '--manifest', str(args.manifest), '--out', str(profile), '--workers', '12'],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    results = {}

    def probe(batch):
        if batch in results:
            return results[batch]
        output = args.out/f'batch_{batch}'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            '-m', module+'probe_training_batch', '--config', str(args.config), '--manifest', str(args.manifest),
            '--model-path', str(args.model_path), '--out', str(output), '--batch-size', str(batch),
            '--stress-profile', str(profile), '--steps', '3']
        print(json.dumps(dict(phase='probe', per_device=batch, global_batch=4*batch)), flush=True)
        path = args.out/f'batch_{batch}.log'
        with path.open('w') as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
        if code:
            if 'out of memory' not in path.read_text().lower():
                raise RuntimeError(f'Non-OOM probe failure; see {path}')
            result = dict(per_device=batch, global_batch=4*batch, fits=False, safe=False)
        else:
            ranks = [json.loads((output/f'rank_{i}.json').read_text()) for i in range(4)]
            result = dict(per_device=batch, global_batch=4*batch, fits=True,
                safe=all(r['reserved_gib'] < .9*r['total_gib'] for r in ranks),
                peak_gib=max(r['peak_gib'] for r in ranks), reserved_gib=max(r['reserved_gib'] for r in ranks),
                seconds=max(r['seconds'] for r in ranks),
                samples_per_second=4*batch*3/max(r['seconds'] for r in ranks))
        results[batch] = result
        (args.out/'progress.json').write_text(json.dumps(list(results.values()), indent=2)+'\n')
        print(json.dumps(result), flush=True)
        return result

    lower = 4
    if not probe(lower)['fits']:
        raise RuntimeError('Mixed data extrema do not fit even at total batch 16')
    upper = lower * 2
    while probe(upper)['fits']:
        lower, upper = upper, upper*2
    while upper-lower > 1:
        middle = (lower+upper)//2
        if probe(middle)['fits']:
            lower = middle
        else:
            upper = middle
    maximum = lower
    safe_values = [b for b,r in results.items() if r['safe']]
    if not safe_values:
        raise RuntimeError('No tested batch has the requested allocator headroom')
    lower, upper = max(safe_values), maximum+1
    while upper-lower > 1:
        middle = (lower+upper)//2
        if probe(middle)['safe']:
            lower = middle
        else:
            upper = middle
    summary = dict(maximum_per_device=maximum, maximum_global=4*maximum,
        recommended_per_device=lower, recommended_global=4*lower,
        first_oom_per_device=maximum+1, allocator_headroom_fraction=.1,
        tests=sorted(results.values(), key=lambda r:r['per_device']),
        profile=str(profile), caveat='Measured on three updates with simultaneous dataset padding maxima, not a full-epoch guarantee.')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
