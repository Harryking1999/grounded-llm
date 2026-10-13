"""Shared physical states, prompts and map/no-map rendering."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from experiments.gcml_counterexamples.src import blocks as rules
from experiments.llm_task_sft.src.protocol import legal_actions, action_key, reason, STOP_REASON
from experiments.llm_task_sft.src.files import atomic_json
from experiments.flamingo_map_reader.src.blocks import blocks_step
from experiments.flamingo_map_reader.src.blocks_prompt import initial_prompt, turn_prompt
from experiments.flamingo_map_reader.src.sft import Demonstration, SupervisedTurn, _ranking
from experiments.flamingo_map_reader.src.graph import timeline_maps

FORMAT = '''Response format:
Output a short factual line beginning with "Reason:", then CONTROL.
For a nonterminal board, choose a legal candidate; state the cells removed, whether goal cells are preserved, and how the removal advances a valid completion.
Use: Reason: Clearing CELLS is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
If that action reaches the goal, use: Reason: Clearing CELLS is legal and leaves exactly the goal board.
At the goal, output: Reason: The current board exactly matches the goal; no further removal is needed.
The final line is <action>ID</action> or, at the goal, <done/>.
ORDERING'''

def prompt(state, goal, executed, first=False, start=None, with_map=False):
    action_ids = [rules.BY_ACTION[action_key(a)] for a in legal_actions(state)]
    step = SimpleNamespace(current=state, candidate_actions=action_ids)
    update = turn_prompt(step, [rules.BY_ACTION[action_key(a)] for a in executed])
    if not first:
        return update
    ordering = ('Before the action line, output "Map-distance ranking to the goal, closest to farthest: RANKING." '
                'Rank the nearest 10 candidates plus current using < and =; input contains every legal candidate.'
                if with_map else 'Do not output candidate rankings.')
    return initial_prompt(state if start is None else start, goal)+'\n\n'+FORMAT.replace('CONTROL','the requested lines').replace('ORDERING',ordering)+'\n\n'+update

def answer(step, candidate=None, with_map=False):
    if candidate is None:
        return STOP_REASON+'\n<done/>'
    action = legal_actions(step.current)[candidate-1]
    text = reason(step.current, step.goal, action)
    if with_map:
        text += '\nMap-distance ranking to the goal, closest to farthest: '+_ranking(step,10)+'.'
    return text+f'\n<action>{candidate}</action>'

def sorted_step(qmap, state, goal):
    # successors() follows the official shape/row/column placement catalogue.
    step = blocks_step(qmap,state,goal)
    expected = tuple(rules.BY_ACTION[action_key(a)] for a in legal_actions(state))
    if step.candidate_actions != expected:
        raise ValueError('Physical candidate order differs from sorted protocol')
    return step

def demonstration(record, qmap, with_map):
    state, goal = int(record['start']),int(record['goal'])
    executed, path, turns = [],[state],[]
    for candidate in [*record['route_ids'],None]:
        step=sorted_step(qmap,state,goal)
        user=prompt(state,goal,executed,not turns,int(record['start']),with_map)
        turns.append(SupervisedTurn(user,answer(step,candidate,with_map),step,tuple(path),candidate))
        if candidate is not None:
            action, state=step.execute(candidate)
            executed.append(dict(rules.PLACEMENTS[action][1]));path.append(state)
    if state != goal:
        raise ValueError('Incomplete training trajectory')
    return Demonstration(tuple(turns),tuple(path),True)

def pseudo_timeline(timeline):
    """Nonzero deterministic slot vectors without state/goal information."""
    batch=timeline.snapshots
    slot=torch.arange(batch.vectors.shape[1],device=batch.vectors.device)[:,None]+1
    dim=torch.arange(batch.vectors.shape[2],device=batch.vectors.device)[None,:]+1
    vectors=torch.sin(slot*dim*.017).unsqueeze(0).expand(batch.vectors.shape[0],-1,-1).clone()
    return replace(timeline,snapshots=replace(batch,vectors=vectors))
