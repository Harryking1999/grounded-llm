"""Deterministic board/length-stratified subset of the accepted physical routes."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from grounded_llm.tasks.blocks.sft import atomic_json


def select_rows(rows, seed):
    rng = random.Random(seed)
    boards = defaultdict(list)
    for row in rows:
        boards[row['board_row']].append(row)
    if len(rows) != 50000 or len(boards) != 1000 or any(len(v) != 50 for v in boards.values()):
        raise ValueError('Expected the accepted 1000-board, 50000-route source')
    done_boards = set(rng.sample(sorted(boards), 200))
    chosen = []
    for board, records in sorted(boards.items()):
        done = [r for r in records if r['shortest_moves'] == 0]
        if len(done) != 1:
            raise ValueError('Expected one terminal route per board')
        quota = 9 if board in done_boards else 10
        groups = defaultdict(list)
        for r in records:
            if r['shortest_moves']:
                groups[r['shortest_moves']].append(r)
        ideal = {k: quota * len(v) / 49 for k, v in groups.items()}
        counts = {k: int(v) for k, v in ideal.items()}
        order = list(groups)
        rng.shuffle(order)
        order.sort(key=lambda k: ideal[k] - counts[k], reverse=True)
        for k in order[:quota - sum(counts.values())]:
            counts[k] += 1
        for k in sorted(groups):
            chosen.extend(rng.sample(groups[k], counts[k]))
        if board in done_boards:
            chosen.extend(done)
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    meta = json.loads((args.source / 'dataset.json').read_text())
    rows = select_rows(meta['rows'], cfg['seed'])
    if args.out.exists():
        raise FileExistsError(args.out)
    (args.out / 'trajectories').mkdir(parents=True)
    for row in rows:
        src = (args.source / 'trajectories' / row['prepared_file']).resolve()
        (args.out / 'trajectories' / row['prepared_file']).symlink_to(src)
    atomic_json(args.out / 'dataset.json', dict(meta, config=cfg, rows=rows,
                source_data=str(args.source), subset='board_length_stratified_10k'))
    atomic_json(args.out / 'distribution.json', dict(samples=len(rows), boards=1000,
        selected_lengths=dict(Counter(r['shortest_moves'] for r in rows)),
        source_lengths=dict(Counter(r['shortest_moves'] for r in meta['rows'])),
        selected_goal_cells=dict(Counter(int(r['goal']).bit_count() for r in rows)),
        source_goal_cells=dict(Counter(int(r['goal']).bit_count() for r in meta['rows']))))
    atomic_json(args.out / 'ready.json', dict(samples=len(rows), smoke=False,
                response_order=meta['response_order'], source_data=str(args.source)))


if __name__ == '__main__':
    main()
