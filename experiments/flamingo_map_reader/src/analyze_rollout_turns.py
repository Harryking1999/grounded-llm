"""Read completed blocks validation cases; no model generation or raw-file writes."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median


def candidate_bin(count):
    for upper in (10, 20, 40, 60):
        if count <= upper:
            return {10: '1-10', 20: '11-20', 40: '21-40', 60: '41-60'}[upper]
    return '61+'


def summarize(rows):
    decisions = [r for r in rows if not r['done']]
    live = [r for r in decisions if r['remaining_shortest'] >= 0]
    required = ('action_keeps_goal_reachable', 'reachable_candidates', 'candidate_slots')
    if any(any(k not in r for k in required) for r in live):
        raise ValueError('Replay omitted live decision metrics')
    n = len(live)
    kept = sum(bool(r['action_keeps_goal_reachable']) for r in live)
    counts = [r['candidates'] for r in live]
    return dict(decisions=len(decisions), solvable=n, already_dead=len(decisions)-n,
                kept=kept, rate=kept/n if n else None,
                shortest=sum(bool(r.get('action_environment_shortest')) for r in live),
                map_minimum_all_decisions_rate=sum(bool(r.get('action_map_minimum')) for r in decisions)/len(decisions) if decisions else None,
                candidates_mean=mean(counts) if n else None,
                candidates_median=median(counts) if n else None,
                random_rate=mean(r['reachable_candidates']/r['candidate_slots']
                                 if r['candidate_slots'] else 0 for r in live) if n else None,
                map_minimum=sum(bool(r.get('action_map_minimum')) for r in live)/n if n else None,
                legal_dead_end=sum(bool(r.get('legal_action')) and not r['action_keeps_goal_reachable'] for r in live),
                invalid_or_stop=sum(not r.get('legal_action', False) for r in live))


def analyze(cases):
    turns, bins, crossed, distances = (defaultdict(list) for _ in range(4))
    first_dead = Counter()
    rows = []
    for case in cases:
        poisoned = False
        for row in case['turns']:
            rows.append(row)
            if row['done']:
                continue
            turn = row['turn'] + 1
            turns[turn].append(row)
            if row['remaining_shortest'] < 0:
                continue
            bucket = candidate_bin(row['candidates'])
            bins[bucket].append(row)
            stage = '1' if turn == 1 else ('2-3' if turn <= 3 else '4+')
            crossed[stage + '/' + bucket].append(row)
            distance = row['remaining_shortest']
            distances[('1-3' if distance <= 3 else ('4-6' if distance <= 6 else '7+')) + '/' + bucket].append(row)
            # Only an executed legal removal can make the next state irreversibly dead.
            if case.get('mode') == 'rollout' and not poisoned and row.get('legal_action') and 'chosen_action' in row and not row['action_keeps_goal_reachable']:
                first_dead[turn] += 1
                poisoned = True
    rollout = all(c.get('mode') == 'rollout' for c in cases)
    nonzero = [c for c in cases if c.get('shortest_moves', 0) > 0]
    return dict(cases=len(cases), nonzero_cases=len(nonzero) if rollout else None,
                reached=sum(c.get('reached_goal', False) for c in cases) if rollout else None,
                nonzero_reached=sum(c.get('reached_goal', False) for c in nonzero) if rollout else None,
                reached_shortest=sum(c.get('reached_goal', False) and c['moves']==c['shortest_moves'] for c in cases) if rollout else None,
                overall=summarize(rows),
                by_turn={str(k):summarize(v) for k,v in sorted(turns.items())},
                by_candidates={k:summarize(v) for k,v in sorted(bins.items())},
                by_stage_candidates={k:summarize(v) for k,v in sorted(crossed.items())},
                by_distance_candidates={k:summarize(v) for k,v in sorted(distances.items())},
                first_irreversible_dead_end_by_turn={str(k):v for k,v in sorted(first_dead.items())})


def main():
    from experiments.flamingo_map_reader.src.collect_evidence import complete_rollout_reachability, load
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs='+', default=['checkpoint-147000', 'checkpoint-196000', 'final'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_root/'blocks/data/manifest.json').read_text())
    expected = {r['trajectory_id'] for r in manifest['records'] if r['split']=='validation'}
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  run=args.run_root.name, split='validation', numbering_variants=1,
                  definition='Nonterminal decisions whose state is solvable before answering; invalid choices count as failures. Turns are one-based.',
                  checkpoints={})
    for checkpoint in args.checkpoints:
        modes = {}
        case_sets = {}
        for mode in ['rollout', 'reference']:
            directory = args.run_root/'evaluation/blocks'/checkpoint/('validation_'+mode+'_map')
            cases = load(directory)
            assert all(c['mode'] == mode and not c['no_map'] for c in cases)
            identities = [(c['trajectory_id'],c['variant']) for c in cases]
            if len(set(identities)) != len(cases) or set(identities) != {(key,0) for key in expected}:
                raise ValueError(f'{checkpoint}/{mode}: incomplete or duplicated validation cases')
            restored = complete_rollout_reachability(cases, manifest)
            modes[mode] = analyze(cases)
            modes[mode]['replayed_missing_turns'] = restored
            modes[mode]['sources'] = [p.parent.name for p in sorted(directory.glob('*/summary.json'))]
            case_sets[mode] = {c['trajectory_id']:c for c in cases}
        paired, identical = 0, 0
        for key in sorted(expected):
            a,b = [case_sets[mode][key]['turns'][0] for mode in ['rollout','reference']]
            if a['done']:
                continue
            assert all(a[k]==b[k] for k in ['current','goal','candidates','remaining_shortest'])
            paired += 1
            identical += a.get('chosen_action') is not None and a.get('chosen_action') == b.get('chosen_action')
        ready=json.loads((args.run_root/'blocks/training/models'/checkpoint/'evaluation_ready.json').read_text())
        result['checkpoints'][checkpoint] = dict(epoch=ready['epoch'], modes=modes,
                                                matched_initial_states=paired, identical_initial_physical_actions=identical)
        result['checkpoints'][checkpoint]['reference_rate_with_rollout_turn_weights'] = sum(
            row['solvable']*modes['reference']['by_turn'][turn]['rate']
            for turn,row in modes['rollout']['by_turn'].items() if row['solvable']
        )/modes['rollout']['overall']['solvable']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(str(args.out))


if __name__ == '__main__':
    main()
