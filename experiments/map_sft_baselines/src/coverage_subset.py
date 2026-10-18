"""Select cached routes by marginal state/goal coverage within length quotas."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

from .common import atomic_json


def decision_pairs(row):
    return {(int(state), int(goal)) for state, goal in row['pairs']}


def select_coverage(rows, samples, seed):
    boards = defaultdict(list)
    for row in rows:
        boards[row['board_row']].append(row)
    if len(rows) != 50000 or len(boards) != 1000 or samples not in (10000, 30000):
        raise ValueError('Expected the accepted 50k source and a 10k or 30k quota')
    rng = random.Random(seed)
    terminal_boards = set(rng.sample(sorted(boards), samples // 50))
    chosen = []
    for board, records in sorted(boards.items()):
        terminal = [r for r in records if not r['shortest_moves']]
        if len(records) != 50 or len(terminal) != 1:
            raise ValueError('Expected 49 nonterminal and one terminal route per board')
        groups = defaultdict(list)
        for row in records:
            if row['shortest_moves']:
                groups[row['shortest_moves']].append(row)
        quota = samples // len(boards) - int(board in terminal_boards)
        ideal = {length: quota * len(group) / 49 for length, group in groups.items()}
        counts = {length: int(value) for length, value in ideal.items()}
        order = list(groups)
        rng.shuffle(order)
        order.sort(key=lambda length: ideal[length] - counts[length], reverse=True)
        for length in order[:quota - sum(counts.values())]:
            counts[length] += 1
        pool = [r for group in groups.values() for r in group]
        rng.shuffle(pool)
        pairs = {r['trajectory_id']: decision_pairs(r) for r in pool}
        seen = set()
        for _ in range(quota):
            eligible = [r for r in pool if counts[r['shortest_moves']] > 0]
            best = max(eligible, key=lambda r: len(pairs[r['trajectory_id']] - seen))
            chosen.append(best)
            seen.update(pairs[best['trajectory_id']])
            counts[best['shortest_moves']] -= 1
            pool.remove(best)
        if board in terminal_boards:
            chosen.extend(terminal)
    rng.shuffle(chosen)
    return chosen


def describe(rows, source_pairs):
    pairs = set().union(*(decision_pairs(row) for row in rows))
    decisions = sum(len(row['pairs']) for row in rows)
    return dict(samples=len(rows), boards=len({r['board_row'] for r in rows}),
                unique_state_goal_pairs=len(pairs), nonterminal_decisions=decisions,
                source_pair_fraction=len(pairs) / len(source_pairs),
                repeated_decision_fraction=1 - len(pairs) / max(1, decisions),
                lengths=dict(Counter(r['shortest_moves'] for r in rows)),
                input_tokens=sum(r.get('map_tokens', 0) for r in rows))


def main():
    parser = argparse.ArgumentParser()
    for arg in ('source', 'original', 'out', 'config'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    policy = cfg['coverage_selection']
    meta = json.loads((args.source / 'dataset.json').read_text())
    rows = meta['rows']
    source_pairs = set().union(*(decision_pairs(row) for row in rows))
    original = json.loads((args.original / 'dataset.json').read_text())['rows']
    candidates = {n: select_coverage(rows, n, cfg['seed']) for n in (10000, 30000)}
    summaries = {str(n): describe(selected, source_pairs) for n, selected in candidates.items()}
    selected_n = (10000 if summaries['10000']['source_pair_fraction'] >=
                  policy['minimum_source_pair_fraction'] else 30000)
    report = dict(source=describe(rows, source_pairs), original_10k=describe(original, source_pairs),
                  candidates=summaries, selected_samples=selected_n,
                  criterion=policy, criterion_is_heuristic=True)
    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / 'coverage_report.json', report)
    selected = candidates[selected_n]
    trajectories = args.out / 'trajectories'
    trajectories.mkdir()
    for row in selected:
        (trajectories / row['prepared_file']).symlink_to(
            (args.source / 'trajectories' / row['prepared_file']).resolve())
    atomic_json(args.out / 'dataset.json', dict(meta, rows=selected, source_data=str(args.source),
                subset='marginal_state_goal_coverage_with_length_quotas', config=cfg))
    atomic_json(args.out / 'ready.json', dict(samples=selected_n, smoke=False,
                response_order=meta['response_order'], source_data=str(args.source)))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
