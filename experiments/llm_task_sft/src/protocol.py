"""Factual prompts, certified shortest routes, and two-line control responses."""

import random
import re
from types import SimpleNamespace

from experiments.gcml_counterexamples.src import blocks as rules
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.flamingo_map_reader.src.blocks_prompt import initial_prompt, turn_prompt
from .files import atomic_json


RESPONSE_INSTRUCTION = '''Response format:
Output exactly two lines: a short reason, then one control tag.
If the current board exactly matches the goal, explain this in one sentence beginning with "Reason:", then output <done/> on the second line.
Otherwise, choose one legal candidate from the current update. On the first line, beginning with "Reason:", state which cells it removes, whether goal cells are preserved, and how it advances a valid completion. On the second line, output <action>ID</action>.
Use one of these reason sentences. Replace CELLS with the removed occupied cells as (row,col), ordered by row then column and joined by " and ".
If the removal leaves a board that still needs solving: Reason: Clearing CELLS is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
If the removal reaches the goal: Reason: Clearing CELLS is legal and leaves exactly the goal board.
If already at the goal: Reason: The current board exactly matches the goal; no further removal is needed.
Do not output candidate rankings or a multi-step plan.'''
STOP_REASON = 'Reason: The current board exactly matches the goal; no further removal is needed.'
CONTINUE_SUFFIX = ' is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.'
ARRIVAL_SUFFIX = ' is legal and leaves exactly the goal board.'


def legal_actions(state):
    return sorted((dict(rules.PLACEMENTS[i][1]) for i in rules.legal_indices(state)),
                  key=lambda action: (action['shape_id'], action['row'], action['col']))


def action_key(action):
    return tuple(action[name] for name in ('shape_id', 'row', 'col'))


def action_from_key(key):
    return dict(zip(('shape_id', 'row', 'col'), key))


def oracle_for(config):
    return DistanceOracle(cache_limit=config['data']['oracle_cache_limit'],
                          seconds=config['data']['oracle_seconds_per_task'])


def sample_routes(start, goal, quota, rng, oracle, attempts_factor=20):
    minimum = oracle.distance(start, goal)
    if minimum < 0:
        raise ValueError('A source task has no certified solution')
    unique, keys = [], set()
    for _ in range(quota * attempts_factor):
        current, route = start, []
        while current != goal:
            distance = oracle.distance(current, goal)
            pool = []
            for action in legal_actions(current):
                after = rules.apply(current, action)
                if after & goal == goal and oracle.distance(after, goal) == distance - 1:
                    pool.append(action)
            if not pool:
                raise AssertionError('Certified nonterminal state lacks an optimal branch')
            chosen = rng.choice(pool)
            route.append(action_key(chosen))
            current = rules.apply(current, chosen)
        key = tuple(route)
        if len(key) != minimum:
            raise AssertionError('Sampled route is not shortest')
        if key not in keys:
            keys.add(key)
            unique.append(key)
        if len(unique) == quota:
            break
    rows = [(route, False) for route in unique]
    while len(rows) < quota:
        rows.append((rng.choice(unique), True))
    return rows, len(unique), minimum


def reason(state, goal, action=None):
    if action is None:
        if state != goal:
            raise ValueError('Stop supervision requires the exact goal')
        return STOP_REASON
    after = rules.apply(state, action)
    if after & goal != goal:
        raise ValueError('Supervised removal damages goal cells')
    removed = ' and '.join(f'({cell // rules.SIZE},{cell % rules.SIZE})'
                           for cell in rules.cells(state ^ after))
    return 'Reason: Clearing ' + removed + (ARRIVAL_SUFFIX if after == goal else CONTINUE_SUFFIX)


def response(state, goal, candidate_id=None):
    if candidate_id is None:
        return reason(state, goal) + '\n<done/>'
    action = legal_actions(state)[candidate_id - 1]
    return reason(state, goal, action) + f'\n<action>{candidate_id}</action>'


def user_update(state, goal, executed, *, first=False, start=None):
    step = SimpleNamespace(current=state, candidate_actions=[rules.BY_ACTION[action_key(a)] for a in legal_actions(state)])
    update = turn_prompt(step, [rules.BY_ACTION[action_key(a)] for a in executed])
    if first:
        return initial_prompt(state if start is None else start, goal) + '\n\n' + RESPONSE_INSTRUCTION + '\n\n' + update
    return update


def route_messages(start, goal, route, oracle):
    current, executed, messages, pairs = start, [], [], []
    minimum = oracle.distance(start, goal)
    for key in route:
        if current == goal:
            raise ValueError('Route continues after reaching the goal')
        pairs.append((current, goal))
        action = action_from_key(key)
        candidate_id = legal_actions(current).index(action) + 1
        messages.append(dict(role='user', content=user_update(current, goal, executed, first=not messages, start=start)))
        messages.append(dict(role='assistant', content=response(current, goal, candidate_id)))
        after = rules.apply(current, action)
        if oracle.distance(after, goal) != oracle.distance(current, goal) - 1:
            raise ValueError('Route or explanation violates shortest-distance progress')
        executed.append(action)
        current = after
    if current != goal or len(route) != minimum:
        raise ValueError('Route is not a complete shortest solution')
    messages.append(dict(role='user', content=user_update(current, goal, executed, first=not messages, start=start)))
    messages.append(dict(role='assistant', content=response(current, goal)))
    return messages, pairs


def encode_messages(messages, tokenizer):
    """Tokenize once; verify all supervised spans against the rendered chat template."""
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans, cursor = [], 0
    for message in messages:
        if message['role'] != 'assistant':
            continue
        segment = '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'
        position = rendered.find(segment, cursor)
        if position < 0:
            raise ValueError('This encoder requires the verified Qwen chat template')
        left = position + len('<|im_start|>assistant\n')
        right = position + len(segment)
        spans.append((left, right))
        cursor = right
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    labels, span_index = [], 0
    for token, (left, right) in zip(encoded['input_ids'], encoded['offset_mapping']):
        while span_index < len(spans) and left >= spans[span_index][1]:
            span_index += 1
        supervised = span_index < len(spans) and left >= spans[span_index][0] and right <= spans[span_index][1] and right > left
        labels.append(token if supervised else -100)
    if not any(label != -100 for label in labels):
        raise ValueError('Trajectory has no supervised tokens')
    return encoded['input_ids'], labels


def parse_control(text, candidate_count):
    tags = re.findall(r'<action>[^<]*</action>|<done\s*/>', text)
    line = text.strip().splitlines()[-1] if text.strip() else ''
    if len(tags) != 1 or len(re.findall(r'<(?:action\b|done\b)',text)) != 1 or tags[0] != line:
        return None, 'control_format'
    if line == '<done/>':
        return None, None
    match = re.fullmatch(r'<action>([1-9][0-9]*)</action>', line)
    if not match:
        return None, 'control_format'
    candidate = int(match.group(1))
    return (candidate, None) if candidate <= candidate_count else (None, 'illegal_action')


def check_reason(text, state, goal, candidate_id, oracle):
    lines = text.strip().splitlines()
    if len(lines) != 2:
        return 'format'
    line = lines[0]
    if candidate_id is None:
        return None if state == goal and line == STOP_REASON else 'fact' if line == STOP_REASON else 'format'
    match = re.fullmatch(r'Reason: Clearing ((?:\([0-9],[0-9]\))(?: and \([0-9],[0-9]\)){1,2})(.*)', line)
    if not match or match.group(2) not in (CONTINUE_SUFFIX, ARRIVAL_SUFFIX):
        return 'format'
    action = legal_actions(state)[candidate_id - 1]
    after = rules.apply(state, action)
    if after & goal != goal:
        return 'fact'
    if line != reason(state, goal, action):
        return 'fact'
    if after != goal:
        before_distance, after_distance = oracle.distance(state, goal), oracle.distance(after, goal)
        if before_distance < 0 or after_distance < 0 or after_distance != before_distance - 1:
            return 'fact'
    return None
