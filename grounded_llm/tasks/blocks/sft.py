"""Shared physical states, prompts and map/no-map rendering."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
import grounded_llm.tasks.blocks.rules as rules
from grounded_llm.tasks.blocks.protocol import legal_actions, action_key, reason, STOP_REASON
from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.blocks.qmap import blocks_step
from grounded_llm.tasks.blocks.prompts import initial_prompt, turn_prompt
from grounded_llm.core.sft.types import Demonstration
from grounded_llm.core.sft.types import SupervisedTurn
from grounded_llm.core.sft.targets import _ranking
from grounded_llm.core.reader.batching import timeline_maps

FORMAT = '''Response format:
Output a short factual line beginning with "Reason:", then CONTROL.
For a nonterminal board, choose a legal candidate; state the cells removed, whether goal cells are preserved, and how the removal advances a valid completion.
Use: Reason: Clearing CELLS is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
If that action reaches the goal, use: Reason: Clearing CELLS is legal and leaves exactly the goal board.
At the goal, output: Reason: The current board exactly matches the goal; no further removal is needed.
The final line is <action>ID</action> or, at the goal, <done/>.
ORDERING'''

def prompt(state, goal, executed, first=False, start=None, with_map=False, response_order='reason_first'):
    action_ids = [rules.BY_ACTION[action_key(a)] for a in legal_actions(state)]
    step = SimpleNamespace(current=state, candidate_actions=action_ids)
    update = turn_prompt(step, [rules.BY_ACTION[action_key(a)] for a in executed])
    if not first:
        return update
    ordering = ('Before the action line, output "Map-distance ranking to the goal, closest to farthest: RANKING." '
                'Rank the nearest 10 candidates plus current using < and =; input contains every legal candidate.'
                if with_map else 'Do not output candidate rankings.')
    formatting=FORMAT.replace('CONTROL','the requested lines').replace('ORDERING',ordering)
    if with_map and response_order=='ranking_first':
        formatting=formatting.replace('Output a short factual line beginning with "Reason:", then the requested lines.',
            'First output the map-distance ranking. Next output a short factual line beginning with "Reason:". Finally output the action line.')
    elif response_order!='reason_first':
        raise ValueError('Unsupported response order')
    return initial_prompt(state if start is None else start, goal)+'\n\n'+formatting+'\n\n'+update

def answer(step, candidate=None, with_map=False, response_order='reason_first'):
    if candidate is None:
        return STOP_REASON+'\n<done/>'
    action = (dict(rules.PLACEMENTS[step.candidate_actions[candidate-1]][1])
              if hasattr(step, 'candidate_actions') else legal_actions(step.current)[candidate-1])
    text = reason(step.current, step.goal, action)
    if with_map:
        ranking='Map-distance ranking to the goal, closest to farthest: '+_ranking(step,10)+'.'
        text = ranking+'\n'+text if response_order=='ranking_first' else text+'\n'+ranking
    return text+f'\n<action>{candidate}</action>'

def sorted_step(qmap, state, goal):
    # successors() follows the official shape/row/column placement catalogue.
    step = blocks_step(qmap,state,goal)
    expected = tuple(rules.BY_ACTION[action_key(a)] for a in legal_actions(state))
    if step.candidate_actions != expected:
        raise ValueError('Physical candidate order differs from sorted protocol')
    return step

def demonstration(record, qmap, with_map, response_order='reason_first', cached_steps=None):
    state, goal = int(record['start']),int(record['goal'])
    executed, path, turns = [],[state],[]
    for candidate in [*record['route_ids'],None]:
        step=sorted_step(qmap,state,goal) if cached_steps is None else cached_steps[len(turns)]
        if step.current!=state or step.goal!=goal:raise ValueError('Cached map differs from physical trajectory')
        user=prompt(state,goal,executed,not turns,int(record['start']),with_map,response_order)
        turns.append(SupervisedTurn(user,answer(step,candidate,with_map,response_order),step,tuple(path),candidate))
        if candidate is not None:
            action, state=step.execute(candidate)
            executed.append(dict(rules.PLACEMENTS[action][1]));path.append(state)
    if state != goal:
        raise ValueError('Incomplete training trajectory')
    return Demonstration(tuple(turns),tuple(path),True)


def renumber_trajectory(record, steps, seed, epoch, index, response_order='ranking_first'):
    """Rebind IDs without changing any physical move, reason or Q content.

    Sample identity and epoch determine the plan independently of rank, worker
    scheduling and resume position. No additional physical samples are created.
    """
    from grounded_llm.core.reader.numbering import numbered_step
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
    demo = demonstration(record, None, True, response_order, cached_steps=steps)
    turns, executed = [], []
    for turn in demo.turns:
        step = numbered_step(turn.step, tuple(map(int, rng.permutation(len(turn.step.candidate_actions)))))
        chosen = None
        if turn.chosen_id is not None:
            physical = turn.step.candidate_actions[turn.chosen_id-1]
            chosen = step.candidate_actions.index(physical)+1
        update = turn_prompt(step, executed)
        if not turns:
            update = turn.user_text.split('[Environment update]', 1)[0]+update
        turns.append(replace(turn, step=step, chosen_id=chosen, user_text=update,
                             answer_text=answer(step, chosen, True, response_order)))
        if chosen is not None:
            executed.append(step.candidate_actions[chosen-1])
    return replace(demo, turns=tuple(turns))


from grounded_llm.core.reader.batching import pseudo_timeline
