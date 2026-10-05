"""Final-checkpoint paired reference Q reversal and assisted-prefix rollouts."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .counterfactual import reverse_candidate_q
from . import blocks_prompt
from .text import chat_ids
from .train import build_reader, pinned_supervision
from .trajectory_dataset import load_record
from .trajectory_eval import GenerationSession, TaskEnvironment, closed_loop, aggregate
from .trajectory_metrics import score_turn, summarize_turns


def append_history(session, user, step, answer):
    """Attach a completed gold turn without generating on excluded states."""
    session.steps.append(step)
    session.messages.append(dict(role='user', content=user))
    session.prefix = chat_ids(session.tokenizer, session.messages, add_generation_prompt=True,
                              **session.config.get('chat_template_kwargs', {}))
    if session.prefix[:len(session.previous)] != session.previous:
        raise ValueError('Historical chat prefix changed')
    session.ids = session.old_ids + [len(session.steps) - 1] * (len(session.prefix) - len(session.previous))
    session.accept(answer)


def scored_answer(step, answer, usage, environment, path, remaining, reachable):
    row = dict(score_turn(step, answer, reported=environment.config['data'].get('reported_candidates')),
               **usage, answer=answer, remaining_shortest=remaining,
               reachable_candidates=reachable, candidate_slots=len(step.candidate_actions))
    chosen = row.get('chosen_id')
    row['action_keeps_goal_reachable'] = environment.chosen_keeps_reachable(step, chosen, path, remaining)
    row['action_environment_shortest'] = environment.chosen_is_shortest(step, chosen, path, remaining)
    row['chosen_action'] = step.candidate_actions[chosen - 1] if row['legal_action'] else None
    return row


def reversed_reference(reverse, demo, environment, baseline):
    rows, excluded = [], dict(terminal=0, already_dead=0)
    if len(baseline['turns']) != len(demo.turns) or baseline['variant'] != 0:
        raise ValueError('Reference baseline has different turns or numbering')
    for index, turn in enumerate(demo.turns):
        step = turn.step
        remaining = environment.remaining(step.current, step.goal, turn.executed_path)
        saved = baseline['turns'][index]
        if int(saved['current']) != step.current or int(saved['goal']) != step.goal or saved['remaining_shortest'] != remaining or saved['candidates'] != len(step.candidate_actions):
            raise ValueError('Saved reference baseline does not match this state')
        if step.done or remaining < 0:
            excluded['terminal' if step.done else 'already_dead'] += 1
            append_history(reverse, turn.user_text, step, turn.answer_text)
            continue
        altered, sources = reverse_candidate_q(step)
        reachable = environment.reachable_candidates(step, turn.executed_path, remaining)
        answer = saved['answer']
        usage = {key: saved.get(key, default) for key, default in
                 (('generated_tokens', 0), ('seconds', 0.), ('context_exhausted', False))}
        other, other_usage = reverse.ask(turn.user_text, altered)
        original_score = scored_answer(step, answer, usage, environment, turn.executed_path, remaining, reachable)
        reverse_score = scored_answer(step, other, other_usage, environment, turn.executed_path, remaining, reachable)
        presented_score = score_turn(altered, other, reported=environment.config['data'].get('reported_candidates'))
        changed = any(not np.isclose(a, b, rtol=1e-10, atol=1e-12)
                      for a, b in zip(step.candidate_map_distances, altered.candidate_map_distances))
        rows.append(dict(turn=index, current=str(step.current), goal=str(step.goal),
                         candidates=len(step.candidate_actions), remaining_shortest=remaining,
                         distance_order_changed=changed, source_candidate_ids=[i + 1 for i in sources],
                         original_minimal_ids=step.map_minimal_candidates,
                         reversed_minimal_ids=altered.map_minimal_candidates,
                         normal=original_score, reverse=reverse_score,
                         reverse_presented_map_minimum=presented_score.get('action_map_minimum', False),
                         reverse_presented_exact_ranking=presented_score.get('exact_ranking', False),
                         changed_physical_action=original_score['chosen_action'] != reverse_score['chosen_action']))
        # Restore the historical map before accepting gold: only the current
        # generation sees the reversal, and no reversed answer enters history.
        reverse.steps[-1] = step
        reverse.accept(turn.answer_text)
    return dict(turns=rows, excluded=excluded)


def safe_choice(environment, step, path):
    remaining = environment.remaining(step.current, step.goal, path)
    candidates = [i for i in range(1, len(step.candidate_actions) + 1)
                  if environment.chosen_is_shortest(step, i, path, remaining)]
    if not candidates:
        raise RuntimeError('No oracle shortest action on an assisted solvable state')
    return min(candidates, key=lambda i: step.candidate_actions[i - 1])


class AssistedSession:
    """Override environment control, but preserve the actual model answer history."""
    def __init__(self, session, environment, helper_steps):
        self.session, self.environment, self.helper_steps = session, environment, helper_steps
        self.path, self.answers = [], []

    def ask(self, user, step):
        self.path.append(step.current)
        answer, usage = self.session.ask(user, step)
        assist = len(self.answers) < self.helper_steps and not step.done
        self.answers.append(dict(model_answer=answer, assisted=assist))
        if assist:
            chosen = safe_choice(self.environment, step, self.path)
            self.answers[-1]['executed_candidate_id'] = chosen
            # This control string is consumed by the existing environment loop;
            # accept() below substitutes the model's own answer in its history.
            return f'<action>{chosen}</action>', usage
        return answer, usage

    def accept(self, answer):
        self.session.accept(self.answers[-1]['model_answer'])


def candidate_bin(count):
    return '1-10' if count <= 10 else '11-20' if count <= 20 else '21-40' if count <= 40 else '41-60' if count <= 60 else '61+'


def paired_summary(cases):
    strata = defaultdict(list)
    for case in cases:
        for row in case['turns']:
            for key in ('all', 'goal/' + case['goal_type'], 'turn/' + str(row['turn'] + 1),
                        'candidates/' + candidate_bin(row['candidates']), 'group/' + case['group'],
                        'distance_changed/' + str(row['distance_order_changed'])):
                strata[key].append(row)
    result = {}
    for key, rows in strata.items():
        result[key] = dict(pairs=len(rows), normal=summarize_turns([r['normal'] for r in rows]),
                           reverse=summarize_turns([r['reverse'] for r in rows]),
                           changed_physical_action=sum(r['changed_physical_action'] for r in rows),
                           reverse_presented_map_minimum=sum(r['reverse_presented_map_minimum'] for r in rows),
                           paired_reachability_delta=sum(int(r['reverse']['action_keeps_goal_reachable']) -
                               int(r['normal']['action_keeps_goal_reachable']) for r in rows))
    return dict(cases=len(cases), strata=result,
                excluded={k: sum(c['excluded'][k] for c in cases) for k in ('terminal', 'already_dead')})


def assisted_summary(cases, manifest):
    summary = aggregate(cases, manifest)
    summary['rollout']['reached_shortest'] = sum(c['reached_goal'] and c['moves'] == c['shortest_moves'] for c in cases)
    summary['model_only_turns'] = summarize_turns([r for c in cases for r in c['turns'] if not r['assisted']])
    summary['goal_types'] = {}
    for goal in sorted({c['goal_type'] for c in cases}):
        selected = [c for c in cases if c['goal_type'] == goal]
        goal_summary = aggregate(selected, manifest)
        goal_summary['rollout']['reached_shortest'] = sum(c['reached_goal'] and c['moves'] == c['shortest_moves'] for c in selected)
        goal_summary['model_only_turns'] = summarize_turns([r for c in selected for r in c['turns'] if not r['assisted']])
        summary['goal_types'][goal] = goal_summary
    return summary


def load_evaluation(manifest_path, checkpoint, model_path):
    manifest = json.loads(manifest_path.read_text())
    config = manifest['config']
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
    expected = pinned_supervision(dict(config=config, manifest=str(manifest_path.resolve()),
        model_source=str(model_path.resolve()),
        map_source=manifest['source_root' if config['task'] == 'graph' else 'q_checkpoint']))
    actual = pinned_supervision(saved['contract'])
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError('Adapter and evaluation contracts differ')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(model_path,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved['adapter'])
    return manifest, config, reader, tokenizer, device, TaskEnvironment(config, manifest)


def main():
    parser = ArgumentParser()
    for name in ('manifest', 'adapter-checkpoint', 'model-path', 'protocol', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--kind', choices=('q_reverse_reference', 'safe_prefix_rollout'), required=True)
    parser.add_argument('--helper-steps', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    task = args.manifest.parent.parent.name
    ready = json.loads(args.adapter_checkpoint.with_name('evaluation_ready.json').read_text())
    if args.adapter_checkpoint.parent.name != 'final' or ready['epoch'] != protocol['final_epochs'][task]:
        raise ValueError('Diagnostic requires the prescribed final checkpoint')
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    manifest, config, reader, tokenizer, device, environment = load_evaluation(
        args.manifest, args.adapter_checkpoint, args.model_path)
    if args.kind == 'safe_prefix_rollout' and (task != 'blocks' or args.helper_steps not in protocol['stages'][1]['helper_steps']):
        raise ValueError('Assisted experiment is blocks rollout with 1, 2, or 3 helper steps')
    records = [r for r in manifest['records'] if r['split'] == protocol['split']][args.start:args.stop]
    baselines = {}
    if args.kind == 'q_reverse_reference':
        run = args.manifest.parents[2]
        group = run / 'evaluation' / task / 'final' / (protocol['split'] + '_reference_map')
        for summary in sorted(group.glob('*/summary.json')):
            bounds = [int(i) for i in summary.parent.name.split('_')]
            if bounds[1] <= args.start or (args.stop is not None and bounds[0] >= args.stop):
                continue
            for line in (summary.parent / 'cases.jsonl').read_text().splitlines():
                case = json.loads(line)
                if case['variant'] == protocol['variant']:
                    if case['trajectory_id'] in baselines:
                        raise ValueError('Duplicate reference baseline')
                    baselines[case['trajectory_id']] = case
        if not all(r['trajectory_id'] in baselines for r in records):
            raise ValueError('Missing completed final reference baseline')
    args.out.mkdir(parents=True)
    results = []
    with (args.out / 'cases.jsonl').open('w') as handle:
        for record in records:
            demo = load_record(args.manifest.parent / 'trajectories', record, config, protocol['variant'])
            session = GenerationSession(reader, tokenizer, config, device)
            if args.kind == 'q_reverse_reference':
                value = reversed_reference(session, demo, environment, baselines[record['trajectory_id']])
            else:
                first = demo.turns[0].user_text.split('[Environment update]', 1)[0] if demo.turns else (
                    blocks_prompt.initial_prompt(int(record['start']), int(record['goal'])))
                assisted = AssistedSession(session, environment, args.helper_steps)
                value = closed_loop(assisted, record, first, environment, config, protocol['variant'])
                for row, answer in zip(value['turns'], assisted.answers):
                    row.update(answer)
            goal_type = ('initial_goal' if record['start'] == record['goal'] else
                         'empty' if int(record['goal']) == 0 else 'nonempty') if task == 'blocks' else 'path'
            result = dict(value, trajectory_id=record['trajectory_id'], group=record['group'],
                          goal_type=goal_type, variant=protocol['variant'], kind=args.kind,
                          mode='reference' if args.kind == 'q_reverse_reference' else 'rollout',
                          helper_steps=args.helper_steps, shortest_moves=record['shortest_moves'],
                          greedy_success=record['greedy_success'])
            results.append(result)
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            print(json.dumps(dict(completed=len(results), trajectory_id=record['trajectory_id'])), flush=True)
    summary = paired_summary(results) if args.kind == 'q_reverse_reference' else assisted_summary(results, manifest)
    summary['contract'] = dict(protocol=protocol, checkpoint=str(args.adapter_checkpoint), ready=ready,
                               kind=args.kind, helper_steps=args.helper_steps, start=args.start, stop=args.stop)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
