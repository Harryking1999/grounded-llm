"""Find data-wide padding extrema for a conservative real-trajectory batch probe."""

from argparse import ArgumentParser
import json
import multiprocessing as mp
from pathlib import Path

import torch


ROOT = None


def profile(record):
    value = torch.load(ROOT / record['prepared_file'], weights_only=True, map_location='cpu')
    lengths = [len(e['input_ids']) for e in value['encodings']]
    variant = max(range(len(lengths)), key=lengths.__getitem__)
    turns = value['demo']['turns']
    return dict(trajectory_id=record['trajectory_id'], variant=variant,
        tokens=lengths[variant], snapshots=len(turns),
        slots=max(t['step']['map_batch']['vectors'].shape[1] for t in turns),
        supervised_tokens=value['encodings'][variant]['answer_tokens'])


def main():
    global ROOT
    parser = ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    ROOT = args.manifest.parent / 'trajectories'
    records = [r for r in manifest['records'] if r['split'] == 'train']
    rows = []
    with mp.get_context('fork').Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap(profile, records, chunksize=8), 1):
            rows.append(row)
            if index % 2000 == 0:
                print(json.dumps(dict(profiled=index, total=len(records))), flush=True)
    extrema = {key: max(rows, key=lambda r: r[key])
               for key in ('tokens', 'snapshots', 'slots', 'supervised_tokens')}
    selected = list({r['trajectory_id']: r for r in extrema.values()}.values())
    result = dict(manifest=str(args.manifest.resolve()), trajectories=len(rows),
        extrema=extrema, selected=selected,
        description='Cycle real examples attaining dataset maxima; mixed padding reaches all T/S/M maxima together.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
