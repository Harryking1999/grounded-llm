"""Own-history rollout and fixed-reference decisions under the same state prompt."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.flamingo_map_reader.src.text import chat_ids

from .protocol import (action_from_key, atomic_json, check_reason, legal_actions,
                       oracle_for, parse_control, response, rules, user_update)


def score_turn(text, state, goal, oracle):
    candidates = legal_actions(state)
    candidate, error = parse_control(text, len(candidates))
    before = oracle.distance(state, goal)
    result = dict(control_error=error, candidate_id=candidate, pre_solvable_nonterminal=before > 0,
                  reachability=False, shortest_progress=False, before_distance=before,
                  reason_error=None, correct_stop=False, after_state=str(state))
    if error:
        return result
    result['reason_error'] = check_reason(text, state, goal, candidate, oracle)
    if candidate is None:
        result['correct_stop'] = state == goal
        if state != goal:
            result['control_error'] = 'premature_stop'
        return result
    action = candidates[candidate-1]
    after = rules.apply(state, action)
    remaining = oracle.distance(after, goal)
    result.update(action=action, after_state=str(after), after_distance=remaining,
                  reachability=before > 0 and remaining >= 0,
                  shortest_progress=before > 0 and remaining == before-1)
    return result


def evaluate_task(record, mode, generate, config):
    start, goal = int(record['start']), int(record['goal'])
    oracle = oracle_for(config)
    state, executed, messages, turns = start, [], [], []
    arrived, stopped, failure = state == goal, False, None
    budget = config['evaluation']['action_budget']
    route = record['reference_actions']
    limit = len(route)+1 if mode == 'reference' else budget+1
    for index in range(limit):
        if mode == 'rollout' and state != goal and not legal_actions(state):
            failure = 'no_legal_actions'
            break
        update = user_update(state, goal, executed, first=index == 0, start=start)
        messages.append(dict(role='user', content=update))
        generated = generate(messages)
        text = generated['text']
        if generated.get('context_overflow'):
            judgement = dict(control_error='context_overflow', candidate_id=None,
                             pre_solvable_nonterminal=oracle.distance(state, goal)>0,
                             reachability=False, shortest_progress=False, reason_error=None,
                             correct_stop=False, after_state=str(state))
        else:
            judgement = score_turn(text, state, goal, oracle)
        turns.append(dict(index=index, before_state=str(state), **generated, **judgement))
        if mode == 'reference':
            if index == len(route):
                stopped = judgement['correct_stop']
                break
            reference_action = action_from_key(route[index])
            candidate_id = legal_actions(state).index(reference_action)+1
            messages.append(dict(role='assistant', content=response(state, goal, candidate_id)))
            executed.append(reference_action)
            state = rules.apply(state, reference_action)
            arrived |= state == goal
            continue
        messages.append(dict(role='assistant', content=text))
        if judgement['control_error']:
            failure = judgement['control_error']
            break
        if judgement['candidate_id'] is None:
            stopped = judgement['correct_stop']
            break
        if len(executed) >= budget:
            failure = 'action_budget'
            break
        executed.append(judgement['action'])
        state = int(judgement['after_state'])
        arrived |= state == goal
    return dict(trajectory_id=record['trajectory_id'], split=record['split'], mode=mode,
                initial_goal=start == goal, initial_pair_exposed=record['sft_initial_pair_exposed'],
                reference_pairs_exposed=record['reference_nonterminal_pairs_exposed'],
                arrived=arrived, arrived_and_stopped=arrived and stopped, failure=failure,
                executed_actions=executed, turns=turns)


def summarize(cases):
    ordinary = [c for c in cases if not c['initial_goal']]
    zero = [c for c in cases if c['initial_goal']]
    eligible = [t for c in ordinary for t in c['turns'] if t['pre_solvable_nonterminal']]
    turns = [t for c in cases for t in c['turns']]
    def rate(numerator, denominator):
        return dict(numerator=numerator, denominator=denominator,
                    rate=numerator/denominator if denominator else None)
    groups = {}
    for exposed in (False, True):
        selected = [c for c in ordinary if c['initial_pair_exposed'] == exposed]
        groups['initial_pair_exposed' if exposed else 'initial_pair_unexposed'] = rate(sum(c['arrived'] for c in selected), len(selected))
    return dict(tasks=len(cases), ordinary_tasks=len(ordinary), initial_goal_tasks=len(zero),decision_turns=len(turns),
        # Reference paths are supplied by the evaluator; their arrival is not model success.
        arrival=rate(sum(c['arrived'] for c in ordinary),len(ordinary)) if cases and cases[0]['mode']=='rollout' else None,
        arrival_and_stop=rate(sum(c['arrived_and_stopped'] for c in ordinary),len(ordinary)) if cases and cases[0]['mode']=='rollout' else None,
        initial_goal_stop=rate(sum(c['arrived_and_stopped'] for c in zero),len(zero)),
        step_reachability=rate(sum(t['reachability'] for t in eligible),len(eligible)),
        shortest_progress=rate(sum(t['shortest_progress'] for t in eligible),len(eligible)),
        control_errors=dict(Counter(t['control_error'] for t in turns if t['control_error'])),
        control_format_error=rate(sum(t['control_error']=='control_format' for t in turns),len(turns)),
        illegal_action=rate(sum(t['control_error']=='illegal_action' for t in turns),len(turns)),
        dead_state_turns=sum(t.get('before_distance',0)<0 for t in turns),
        reason_errors=dict(Counter(t['reason_error'] for t in turns if t['reason_error'])),
        reason_checked_turns=sum(t['reason_error'] is not None or t['control_error'] in (None,'premature_stop') for t in turns),
        generated_limit_hits=sum(t.get('generation_limit_hit',False) for t in turns),
        exposure_groups=groups if cases and cases[0]['mode']=='rollout' else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','data','model-path','out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', choices=('validation','test'), required=True)
    parser.add_argument('--mode', choices=('rollout','reference'), required=True)
    parser.add_argument('--shard-index', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audit = json.loads((args.data/'summary.json').read_text())
    records = [r for r in json.loads((args.data/'evaluation.json').read_text()) if r['split']==args.split]
    size = config['evaluation']['shard_tasks']
    records = records[args.shard_index*size:(args.shard_index+1)*size]
    if not records:
        raise ValueError('Empty shard')
    if (args.out/'summary.json').exists():
        raise FileExistsError('Completed shard already exists')
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation=config['training']['attention'],local_files_only=True).to('cuda').eval()
    model.config.use_cache = True
    limit, context = audit['generation_limit'], audit['native_context_limit']

    @torch.inference_mode()
    def generate(messages):
        prompt = chat_ids(tokenizer, messages, add_generation_prompt=True)
        if len(prompt)+limit > context:
            return dict(text='', prompt_tokens=len(prompt), output_tokens=0, context_overflow=True)
        ids = torch.tensor([prompt],device='cuda')
        answer = model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
            max_new_tokens=limit,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
            use_cache=True)[0,len(prompt):].tolist()
        return dict(text=tokenizer.decode(answer,skip_special_tokens=True),prompt_tokens=len(prompt),
                    output_tokens=len(answer),generation_limit_hit=len(answer)==limit and answer[-1]!=tokenizer.eos_token_id)

    started, cases = time.monotonic(), []
    with (args.out/'cases.jsonl').open('w') as stream:
        for record in records:
            case = evaluate_task(record,args.mode,generate,config)
            cases.append(case); stream.write(json.dumps(case)+'\n'); stream.flush()
    atomic_json(args.out/'summary.json',dict(model_path=str(args.model_path),split=args.split,mode=args.mode,
        shard_index=args.shard_index,generation_limit=limit,seconds=time.monotonic()-started,metrics=summarize(cases)))


if __name__ == '__main__':
    main()
