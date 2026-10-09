"""Read saved three-task outputs, reconstruct ranking scores, and summarize rounds."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from experiments.flamingo_map_reader.src.analyze_blocks_ranking import (
    ndcg, meeting_ranking_scores, MEETING_METRICS)
from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap, blocks_step
from experiments.flamingo_map_reader.src.blocks_checkpoint_queue import checkpoints
from experiments.flamingo_map_reader.src.relations import parse_ranking
from experiments.flamingo_map_reader.src.evaluate_blocks import parse_control
from experiments.flamingo_map_reader.src.trajectory_dataset import load_record
from experiments.flamingo_map_reader.src.trajectory_metrics import still_solvable, termination_counts


def read_cases(directory):
    cases = []
    for path in sorted(directory.glob('*/cases.jsonl')):
        if '.interrupted' in str(path.parent) or not (path.parent/'summary.json').exists():
            continue
        text = path.read_text()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1 or text.endswith('\n'):
                    raise
    keys = [(c['trajectory_id'], c['variant']) for c in cases]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate case in {directory}')
    return cases


def summarize(rows):
    live = [r for r in rows if still_solvable(r)]
    decisions = [r for r in rows if not r['done'] and r.get('candidates', 0) > 0]
    ranked = [r for r in decisions if r.get('ranking_ndcg_at_10') is not None]
    extra = {}
    for key in MEETING_METRICS:
        scored = [r[key] for r in decisions if r.get(key) is not None]
        extra[key] = sum(scored)/len(scored) if scored else None
    return dict(**extra, turns=len(rows), solvable=len(live),
                safe=sum(bool(r.get('action_keeps_goal_reachable')) for r in live),
                random_safe_mean=sum(r.get('reachable_candidates', 0)/r['candidates'] for r in live)/len(live) if live else None,
                shortest=sum(bool(r.get('action_environment_shortest')) for r in live),
                ndcg_n=len(ranked),
                ndcg_mean=sum(r['ranking_ndcg_at_10'] for r in ranked)/len(ranked) if ranked else None,
                ndcg_random_mean=sum(r['random_ndcg_at_10'] for r in ranked)/len(ranked) if ranked else None,
                map_decisions=len(decisions),
                ranking_parseable=sum(r.get('ranking_parseable', False) for r in decisions),
                topk_set_exact=sum(r.get('valid_ranking', False) for r in decisions),
                topk_recall_mean=sum(r.get('topk_recall', 0) for r in decisions)/len(decisions) if decisions else None,
                map_minimum=sum(r.get('action_map_minimum', False) for r in decisions),
                follows_own_top=sum(r.get('follows_own_top', False) for r in decisions),
                selected_in_ranking=sum(r.get('selected_in_ranking', False) for r in decisions),
                legal=sum(r.get('legal_action', False) for r in decisions),
                random_map_minimum_mean=sum(r.get('random_map_minimum', 0) for r in decisions)/len(decisions) if decisions else None,
                **termination_counts(rows))


def add_readout_scores(row, distances, ranking_contract):
    ranks = parse_ranking(row['answer'], row['candidates'], ranking_contract['k'])
    value, chance = ndcg(distances, ranks, k=ranking_contract['k'],
                         gain=ranking_contract['gain'], discount=ranking_contract['discount'])
    try:
        chosen = parse_control(row['answer'])
    except ValueError:
        chosen = None
    candidates = {i:r for i,r in (ranks or {}).items() if i > 0}
    expected = set(sorted(range(1, len(distances)+1), key=lambda i:(distances[i-1], i))[:ranking_contract['k']])
    row.update(ranking_ndcg_at_10=value, random_ndcg_at_10=chance,
               ranking_parseable=ranks is not None,
               topk_recall=len(expected & candidates.keys())/len(expected),
               random_map_minimum=sum(np.isclose(d, min(distances), rtol=1e-10, atol=1e-12) for d in distances)/len(distances),
               selected_in_ranking=chosen in candidates,
               follows_own_top=chosen in candidates and candidates[chosen] == min(candidates.values()))
    row.update(meeting_ranking_scores(distances, ranks))
    return {k:row[k] for k in (*MEETING_METRICS, 'ranking_ndcg_at_10', 'random_ndcg_at_10',
                               'ranking_parseable', 'selected_in_ranking', 'follows_own_top', 'random_map_minimum', 'topk_recall')}


def candidate_bucket(n):
    return '1–10' if n <= 10 else '11–20' if n <= 20 else '21–40' if n <= 40 else '41–60' if n <= 60 else '61+'


def summarize_cases(cases):
    rows = [r for c in cases for r in c['turns']]
    rounds = defaultdict(list)
    first_loss = Counter()
    kinds = Counter()
    buckets = defaultdict(list)
    live_buckets = defaultdict(list)
    for r in rows:
        if not r['done'] and r.get('candidates', 0) > 0:
            buckets[candidate_bucket(r['candidates'])].append(r)
            if still_solvable(r):
                live_buckets[candidate_bucket(r['candidates'])].append(r)
    for c in cases:
        for r in c['turns']:
            rounds[r['turn']+1].append(r)
        bad = next((r for r in c['turns'] if still_solvable(r) and
                    not r.get('action_keeps_goal_reachable', False)), None)
        if bad is not None:
            first_loss[bad['turn']+1] += 1
            kinds['legal_fatal_action' if bad['legal_action'] else 'nonaction_or_format'] += 1
    grouped = defaultdict(list)
    for c in cases:
        grouped[c['group']].append(c)
    return dict(cases=len(cases), groups=dict(Counter(c['group'] for c in cases)),
                reached=sum(bool(c.get('reached_goal')) for c in cases),
                success=sum(bool(c.get('success')) for c in cases),
                overall=summarize(rows), solvable_readout=summarize([r for r in rows if still_solvable(r)]),
                first_round=summarize([c['turns'][0] for c in cases]),
                by_round={str(i):summarize(rr) for i,rr in sorted(rounds.items())},
                by_group={g:dict(cases=len(cc), success=sum(bool(c.get('success')) for c in cc),
                                overall=summarize([r for c in cc for r in c['turns']]))
                          for g,cc in grouped.items()},
                by_candidate_count={k:summarize(rr) for k,rr in buckets.items()},
                solvable_by_candidate_count={k:summarize(rr) for k,rr in live_buckets.items()},
                first_loss_by_round=dict(sorted(first_loss.items())), first_loss_kinds=dict(kinds),
                safe_prefix_survival={str(i):len(cases)-sum(n for r,n in first_loss.items() if r<=i)
                                      for i in sorted(rounds)})


def overall_completion(ordinary, initial_goal, failure, weights, expected):
    groups = dict(ordinary=ordinary, initial_goal=initial_goal, failure_context=failure)
    if any(groups[k]['cases'] != expected[k] for k in groups):
        return None
    correct = dict(ordinary=ordinary['success'],
                   initial_goal=initial_goal['overall']['correct_goal_claims'],
                   failure_context=failure['overall']['correct_no_solution'])
    return dict(weighted_rate=sum(weights[k]*correct[k]/expected[k] for k in correct),
                pooled_correct=sum(correct.values()), pooled_tasks=sum(expected.values()),
                correct_by_task=correct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ranking-cache', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/blocks_three_task_evaluation.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract = json.loads(args.config.read_text())
    manifest = json.loads((args.run/'data/manifest.json').read_text())
    records = {r['trajectory_id']:r for r in manifest['records']}
    qmap = FrozenBoardMap.load(Path(manifest['q_checkpoint']))
    demos = {}
    cache = json.loads(args.ranking_cache.read_text()) if args.ranking_cache and args.ranking_cache.exists() else {}
    cache = {k:v for k,v in cache.items() if isinstance(v, dict) and
             'topk_recall' in v and all(name in v for name in MEETING_METRICS)}
    result = dict(run=args.run.name, observed_at_utc=datetime.now(timezone.utc).isoformat(),
                  evaluation_contract=contract, checkpoints={})
    expected = {k:v['tasks'] for k,v in contract['tasks'].items()}
    for checkpoint in sorted(checkpoints(args.run/'training/models'), key=lambda c:c['step']):
        step = checkpoint['step']
        root = args.run/f'evaluation_half_epoch/step-{step}'
        values = {}
        for mode in ('reference', 'rollout'):
            cases = read_cases(root/f'test_{mode}')
            for case in cases:
                rec = records[case['trajectory_id']]
                identity = (rec['trajectory_id'],case['variant'])
                if mode == 'reference' and identity not in demos:
                    demos[identity] = load_record(args.run/'data/trajectories', rec, manifest['config'], case['variant'])
                rng = np.random.default_rng(rec['sample_seed'] + 104729*(case['variant']+1))
                for row in case['turns']:
                    token = f"{mode}:{step}:{identity[0]}:{identity[1]}:{row['turn']}"
                    cached = cache.get(token)
                    # Numbering RNG advances on every rollout turn, even excluded states.
                    snapshot = None
                    if mode == 'rollout' and cached is None:
                        snapshot = blocks_step(qmap, int(row['current']), int(row['goal']), rng=rng)
                    elif mode == 'rollout':
                        rng.permutation(row['candidates'])
                    if row['done'] or not row.get('candidates'):
                        continue
                    if cached is None:
                        if mode == 'reference':
                            snapshot = demos[identity].turns[row['turn']].step
                        assert snapshot.current == int(row['current']) and len(snapshot.candidate_actions) == row['candidates']
                        if row.get('legal_action') and row.get('chosen_action') is not None:
                            assert snapshot.candidate_actions[row['chosen_id']-1] == row['chosen_action']
                        cached = add_readout_scores(row, snapshot.candidate_map_distances, contract['ranking'])
                        cache[token] = cached
                    row.update(cached)
            values[mode] = summarize_cases(cases)
        for task, directory in [('initial_goal', 'initial_goal_reference'),
                                ('failure_context', 'test_no_solution_reference')]:
            cases = read_cases(root/directory)
            for c in cases:
                assert len(c['turns']) == 1
                assert bool(c['turns'][0]['done']) == (task == 'initial_goal')
            values[task] = summarize_cases(cases)
        if contract.get('overall_completion') != 'disabled_report_tasks_separately':
            pooled_rows = []
            for directory in ('test_rollout', 'initial_goal_reference', 'test_no_solution_reference'):
                pooled_rows.extend(r for c in read_cases(root/directory) for r in c['turns'])
            values['overall_termination'] = termination_counts(pooled_rows)
            values['overall_completion'] = overall_completion(values['rollout'],values['initial_goal'],
                values['failure_context'],contract['task_weights'],expected)
        result['checkpoints'][str(step)] = dict(epoch=checkpoint['epoch'], **values)
        print(json.dumps(dict(step=step, ordinary=values['rollout']['cases'],
                             initial_goal=values['initial_goal']['cases'], failure=values['failure_context']['cases'])), flush=True)
    result['observed_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if args.ranking_cache:
        args.ranking_cache.write_text(json.dumps(cache))


if __name__ == '__main__':
    main()
