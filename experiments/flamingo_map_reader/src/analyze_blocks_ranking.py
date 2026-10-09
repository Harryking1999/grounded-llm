"""Offline NDCG@10 of saved final blocks answers; never alters evaluation shards."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def rank_grade_gains(distances, k=10):
    """Top-K ordinal grades K..1, others zero; true ties share mean slot grade."""
    k = min(k, len(distances))
    groups = []
    for i in sorted(range(len(distances)), key=lambda j: (distances[j], j)):
        if groups and math.isclose(distances[i], distances[groups[-1][0]], rel_tol=1e-10, abs_tol=1e-12):
            groups[-1].append(i)
        else:
            groups.append([i])
    gains = [0.0] * len(distances)
    position = 1
    for group in groups:
        grade = sum(max(k + 1 - p, 0) for p in range(position, position + len(group))) / len(group)
        for i in group:
            gains[i] = grade
        position += len(group)
    return gains


def inverse_rank_gains(distances):
    """Reciprocal true positions; equal distances share mean occupied gain."""
    ordered = sorted(range(len(distances)), key=lambda i: distances[i])
    gains = [0.0] * len(distances)
    offset = 0
    while offset < len(ordered):
        stop = offset + 1
        while stop < len(ordered) and math.isclose(distances[ordered[offset]],
                distances[ordered[stop]], rel_tol=1e-10, abs_tol=1e-12):
            stop += 1
        value = sum(1.0 / (i + 1) for i in range(offset, stop)) / (stop - offset)
        for i in ordered[offset:stop]:
            gains[i] = value
        offset = stop
    return gains


def ndcg(distances, ranks, k=10, gain='inverse_distance', discount='log2'):
    """Linear gains, average predicted ties, and zero gain for missing slots."""
    assert all(math.isfinite(d) and d >= 0 for d in distances)
    assert gain in ('inverse_distance', 'rank_grade', 'inverse_rank', 'reciprocal_distance',
                    'minmax_distance')
    if gain == 'rank_grade':
        gains = rank_grade_gains(distances, k)
    elif gain == 'inverse_rank':
        gains = inverse_rank_gains(distances)
    elif gain == 'reciprocal_distance':
        gains = [1.0 / max(d, 1e-12) for d in distances]
    elif gain == 'minmax_distance':
        if not distances:
            return None, None
        nearest, farthest = min(distances), max(distances)
        # Equal distances carry no ordering information; count these states
        # separately when reporting and give every candidate equal gain.
        gains = ([1.0] * len(distances) if math.isclose(nearest, farthest,
                 rel_tol=1e-10, abs_tol=1e-12) else
                 [(farthest - d) / (farthest - nearest) for d in distances])
    else:
        gains = [1.0 / (1.0 + d) for d in distances]
    k = min(k, len(gains))
    if not k:
        return None, None
    assert discount in ('log2', 'reciprocal_rank')
    weights = [1 / (i + 1) if discount == 'reciprocal_rank' else 1 / math.log2(i + 2) for i in range(k)]
    ideal = sum(g * w for g, w in zip(sorted(gains, reverse=True), weights))
    chance = sum(gains) / len(gains) * sum(weights) / ideal
    if ranks is None:
        return 0.0, chance
    groups = defaultdict(list)
    for candidate, rank in ranks.items():
        if candidate:
            assert 1 <= candidate <= len(gains)
            groups[rank].append(gains[candidate - 1])
    dcg, position = 0.0, 0
    for rank in sorted(groups):
        values = groups[rank]
        dcg += sum(values) / len(values) * sum(weights[position:min(position+len(values), k)])
        position += len(values)
        if position >= k:
            break
    value = dcg / ideal
    assert -1e-12 <= value <= 1+1e-12
    return min(1.0, max(0.0, value)), chance


MEETING_METRICS = ('ndcg_inverse_rank_at_10', 'ndcg_minmax_distance_at_10',
                   'ndcg_inverse_rank_at_1', 'ndcg_minmax_distance_at_1')


def meeting_ranking_scores(distances, ranks):
    """Official gains: reciprocal true position and per-state normalized distance."""
    return {name: ndcg(distances, ranks, k=k, gain=gain)[0]
            for name, k, gain in zip(MEETING_METRICS, (10, 10, 1, 1),
                ('inverse_rank', 'minmax_distance', 'inverse_rank', 'minmax_distance'))}


def check_metric():
    assert ndcg([0, 1, 3], {0: 3, 1: 0, 2: 1, 3: 2})[0] == 1.0
    assert ndcg([0, 1], None)[0] == 0.0
    assert ndcg([], None) == (None, None)
    assert ndcg([0, 1], {0: 0}, 2)[0] == 0.0
    expected = (0.5 + 1/math.log2(3))/(1+0.5/math.log2(3))
    assert math.isclose(ndcg([0, 1], {0: 0, 1: 2, 2: 1})[0], expected)
    assert math.isclose(ndcg([0, 1], {0: 0, 1: 1, 2: 1}, 1)[0], .75)
    assert math.isclose(ndcg([0, 1], {0: 0, 1: 1}, 2)[0], 1/(1+.5/math.log2(3)))
    # Independently enumerate tie permutations crossing the cutoff.
    from itertools import permutations
    distances = [0, 1, 2, 3]
    values = []
    for order in permutations(range(1, 5)):
        values.append(ndcg(distances, {0: 0, **{c:i+1 for i,c in enumerate(order)}}, 2)[0])
    tied = ndcg(distances, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}, 2)[0]
    assert math.isclose(tied, sum(values)/len(values))
    assert rank_grade_gains([5, 1, 3, 2], 3) == [0, 3, 1, 2]
    assert rank_grade_gains([0, 0, 2], 2) == [1.5, 1.5, 0]
    assert rank_grade_gains([0, 1, 1], 2) == [2, .5, .5]
    assert rank_grade_gains([1, 1, 1], 2) == [1, 1, 1]
    assert rank_grade_gains([0, 1], 10) == [2, 1]
    assert rank_grade_gains(list(range(9)) + [9, 9, 11], 10)[9:] == [.5, .5, 0]
    for distances, cutoff in [([0, 1, 2, 3], 2), ([0, 1, 1, 3], 2), ([2, 2, 2], 2)]:
        scores = [ndcg(distances, {0: 0, **{c: i+1 for i, c in enumerate(order)}}, cutoff, 'rank_grade')[0]
                  for order in permutations(range(1, len(distances)+1))]
        chance = ndcg(distances, None, cutoff, 'rank_grade')[1]
        assert math.isclose(sum(scores)/len(scores), chance)
        tied = ndcg(distances, {0: 0, **{c: 1 for c in range(1, len(distances)+1)}}, cutoff, 'rank_grade')[0]
        assert math.isclose(tied, chance)
    assert ndcg([0, 1, 2, 3], {0: 0, 1: 1, 2: 2}, 2, 'rank_grade')[0] == 1.0
    assert ndcg([0, 1, 2, 3], {0: 0, 3: 1, 4: 2}, 2, 'rank_grade')[0] == 0.0
    for distances, cutoff in [([0, 1], 10), ([0, 1, 2], 10), ([0, 1, 2, 3], 10), ([0, 1, 1, 3], 2)]:
        scores = [ndcg(distances, {0: 0, **{c: i+1 for i, c in enumerate(order)}}, cutoff, 'rank_grade', 'reciprocal_rank')[0]
                  for order in permutations(range(1, len(distances)+1))]
        chance = ndcg(distances, None, cutoff, 'rank_grade', 'reciprocal_rank')[1]
        assert math.isclose(sum(scores)/len(scores), chance)
        tied = ndcg(distances, {0: 0, **{c: 1 for c in range(1, len(distances)+1)}}, cutoff, 'rank_grade', 'reciprocal_rank')[0]
        assert math.isclose(tied, chance)
    assert math.isclose(ndcg([0, 1], {0: 0, 2: 1, 1: 2}, 10, 'rank_grade', 'reciprocal_rank')[0], .8)
    assert math.isclose(ndcg([0, 1], None, 10, 'rank_grade', 'reciprocal_rank')[1], .9)


def summarize(rows):
    scored = [r for r in rows if r['ndcg_at_10'] is not None]
    valid = [r for r in scored if r['parseable_ranking']]
    def mean(key, subset):
        return sum(r[key] for r in subset)/len(subset) if subset else None
    return dict(decision_turns=len(rows), scored_turns=len(scored),
                no_candidate_turns=len(rows)-len(scored),
                malformed_or_missing_rankings=sum(not r['parseable_ranking'] for r in scored),
                partial_rankings=sum(r['parseable_ranking'] and r['predicted_candidates'] < min(10,r['candidates']) for r in scored),
                mean_ndcg_at_10=mean('ndcg_at_10',scored),
                mean_uniform_random_ndcg_at_10=mean('random_ndcg_at_10',scored),
                mean_ndcg_parseable_only=mean('ndcg_at_10',valid),
                exact_ranking_rate=mean('exact_ranking',scored))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gain', choices=['inverse_distance', 'rank_grade'], default='inverse_distance')
    parser.add_argument('--discount', choices=['log2', 'reciprocal_rank'], default='log2')
    args=parser.parse_args()
    check_metric()
    import numpy as np
    import torch
    from experiments.flamingo_map_reader.src.blocks import FrozenBoardMap, blocks_step
    from experiments.flamingo_map_reader.src.trajectory_dataset import load_record
    from experiments.flamingo_map_reader.src.relations import parse_ranking, score_relationships
    torch.set_num_threads(1)
    run=args.run_root
    data=run/'blocks/data'
    manifest=json.loads((data/'manifest.json').read_text())
    records={r['trajectory_id']:r for r in manifest['records'] if r['split']=='validation'}
    assert len(records)==1100
    assert manifest['config']['data']['reported_candidates']==10
    ready=json.loads((run/'blocks/training/models/final/evaluation_ready.json').read_text())
    qmap=FrozenBoardMap.load(Path(manifest['q_checkpoint']))
    result=dict(run_root=str(run),checkpoint=ready,split='validation',variant=0,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        definitions=dict(metric=('Normalized rank-discounted gain@10 (1/i)' if args.discount=='reciprocal_rank' else 'NDCG@10'),
            gain_scheme=args.gain, discount_scheme=args.discount,
            gain=('K=min(10,N); map ranks 1..K receive linear grades K..1; ranks >K receive 0.'
                  if args.gain == 'rank_grade' else '1/(1+frozen_map_distance)'),
            true_ties=('Map-distance ties (rel_tol=1e-10, abs_tol=1e-12) receive the mean grade of their occupied ordinal slots, including zeros beyond K.'
                       if args.gain == 'rank_grade' else 'Equal map distances give equal gain.'),
            dcg=('sum(gain_at_rank_i/i), i starts at 1; linear gain, no logarithm or exponentiation'
                 if args.discount=='reciprocal_rank' else 'sum(gain_at_rank_i/log2(i+1)), i starts at 1; linear gain (no 2**gain transform)'),
            ideal='Sort all legal successor candidates by map distance, keep min(10,N).',
            current='Exclude current from the ranked retrieval items; only successor candidates are scored.',
            ties='Average gain across each predicted tie group, including cutoff-straddling ties.',
            malformed='Malformed or absent ranking scores 0 when candidates exist.',
            partial='Missing ranks contribute zero; a parseable wrong top-10 set is scored by its actual gains.',
            no_candidates='NDCG undefined, excluded and counted separately.',
            cohorts='all: all nonterminal turns with candidates, including already-unreachable states; solvable: remaining_shortest >= 1.',
            aggregation='Unweighted mean across eligible decision turns; each step uses its actual surviving cases.',
            random='Exact expected NDCG for a uniformly random permutation of all legal candidates; full top-k, no format errors.',
            step='1-based decision number = saved turn + 1'),modes={})
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'per_turn.jsonl').open('w') as detail:
        for mode in ('rollout','reference'):
            directory=run/'evaluation/blocks/final'/f'validation_{mode}_map'
            files=sorted(directory.glob('*/summary.json'))
            cases=[json.loads(l) for f in files for l in (f.parent/'cases.jsonl').read_text().splitlines() if l.strip()]
            identities=[(c['trajectory_id'],c['variant']) for c in cases]
            assert len(identities)==len(records) and set(identities)=={(i,0) for i in records}
            all_rows=[]
            terminal=0
            checks=Counter()
            for ci,case in enumerate(cases):
                assert case['mode']==mode and not case['no_map']
                record=records[case['trajectory_id']]
                if mode=='reference':
                    demo=load_record(data/'trajectories',record,manifest['config'],case['variant'])
                    assert len(demo.turns)==len(case['turns'])
                else:
                    rng=np.random.default_rng(record['sample_seed']+104729*(case['variant']+1))
                for index,row in enumerate(case['turns']):
                    assert row['turn']==index
                    if mode=='reference':
                        step=demo.turns[index].step
                    else:
                        assert row['current']==case['actual_path'][index]
                        step=blocks_step(qmap,int(row['current']),int(row['goal']),rng=rng)
                        if index<len(case['actual_actions']):
                            action,destination=step.execute(row['chosen_id'])
                            assert action==case['actual_actions'][index]
                            assert str(destination)==case['actual_path'][index+1]
                            checks['replayed_actions']+=1
                    assert str(step.current)==row['current'] and str(step.goal)==row['goal']
                    assert step.done==row['done'] and len(step.candidate_actions)==row['candidates']
                    if row['done']:
                        terminal+=1
                        continue
                    if step.candidate_actions:
                        rebuilt=score_relationships(step,row['answer'],10)
                        for key in ('valid_ranking','exact_ranking','pairwise_correct','action_map_minimum'):
                            assert rebuilt[key]==row[key], (mode,case['trajectory_id'],index,key,rebuilt[key],row[key])
                        checks['reproduced_existing_ranking_metrics']+=1
                    ranks=parse_ranking(row['answer'],len(step.candidate_actions),10)
                    value,chance=ndcg(step.candidate_map_distances,ranks,gain=args.gain,discount=args.discount)
                    scored=dict(mode=mode,trajectory_id=case['trajectory_id'],variant=case['variant'],
                        step=index+1,remaining_shortest=row['remaining_shortest'],candidates=row['candidates'],
                        ndcg_at_10=value,random_ndcg_at_10=chance,parseable_ranking=ranks is not None,
                        predicted_candidates=0 if ranks is None else len(ranks)-1,
                        exact_ranking=row.get('exact_ranking',False))
                    all_rows.append(scored)
                    detail.write(json.dumps(scored)+'\n')
                if (ci+1)%100==0:
                    print(json.dumps(dict(mode=mode,cases_done=ci+1,decisions=len(all_rows))),flush=True)
            cohorts={}
            for label,subset in [('all',all_rows),('solvable',[r for r in all_rows if r['remaining_shortest']>=1])]:
                steps=defaultdict(list)
                for row in subset:
                    steps[row['step']].append(row)
                cohorts[label]=dict(overall=summarize(subset),by_step=[dict(step=s,**summarize(rr)) for s,rr in sorted(steps.items())])
            result['modes'][mode]=dict(cases=len(cases),terminal_turns=terminal,checks=dict(checks),
                source_shards=[str(f.parent) for f in files],cohorts=cohorts)
            print(json.dumps(dict(mode=mode,overall=cohorts['all']['overall'],checks=dict(checks))),flush=True)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DONE',str(args.out),flush=True)


if __name__=='__main__':
    main()
