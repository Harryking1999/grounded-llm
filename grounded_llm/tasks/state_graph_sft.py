"""Task state-graph adapters for complete-goal FLAN demonstrations.

Physical rules and exact teachers stay in their task modules. Teachers are used
only to certify data and score answers; map steps contain raw-state Q vectors.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from copy import copy
import json
import numpy as np
import torch

from grounded_llm.core.reader.memory import q_map_batch
from grounded_llm.core.reader.numbering import numbered_step
from grounded_llm.core.sft.types import Demonstration, SupervisedTurn
from grounded_llm.core.sft.targets import _ranking
from grounded_llm.core.sft.ranking import parse_ranking, meeting_ranking_scores
from grounded_llm.tasks.blocks.protocol import parse_control
from grounded_llm.tasks.text import section_text


@dataclass(frozen=True)
class Step:
    current: object
    goal: object
    candidate_actions: tuple
    candidate_destinations: tuple
    candidate_map_distances: tuple
    current_map_distance: float
    map_minimal_candidates: tuple
    map_batch: object
    current_text: str

    @property
    def done(self):
        return self.current == self.goal

    def execute(self, local_id):
        if type(local_id) is not int or not 1 <= local_id <= len(self.candidate_actions):
            raise ValueError('Candidate outside the complete legal list')
        return self.candidate_actions[local_id-1], self.candidate_destinations[local_id-1]


@lru_cache(maxsize=12)
def frozen_reader(task, checkpoint, context, device):
    context = json.loads(context)
    if task == 'blocks':
        from .blocks.qmap import FrozenBoardMap
        return FrozenBoardMap.load(Path(checkpoint), device)
    if task == 'bridge':
        from .bridge.environment import Bridge
        from .bridge.qmap import MapReader
        return MapReader(Bridge(context['times']), checkpoint, device)
    if task == 'token_swapping':
        from .token_swapping.environment import TokenSwapping
        from .token_swapping.qmap import MapReader
        return MapReader(TokenSwapping(10, context['edges']), checkpoint, device)
    if task == 'rush_hour':
        from .rush_hour.environment import RushHour
        from .rush_hour.qmap import MapReader
        return MapReader(RushHour(context['board']), checkpoint, device)
    if task == 'game24':
        from .game24.environment import Game24
        from .game24.qmap import MapReader
        return MapReader(Game24(), checkpoint, device)
    from .restoration_sft import load_map
    return load_map(task, teacher_environment(task, context), checkpoint, device)


@lru_cache(maxsize=8)
def _restoration_environment(task, oracle):
    from .restoration_sft import environment
    return environment(task, Path(oracle))


def teacher_environment(task, context):
    return _restoration_environment(task, context['oracle'])


def action_key(action):
    return action if isinstance(action, str) else json.dumps(action, separators=(',', ':'), sort_keys=True)


@lru_cache(maxsize=64)
def rush_teacher(path):
    from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
    saved=np.load(path)
    return {tuple(map(int,s)):i for i,s in enumerate(saved['states'])},UnitGraphOracle(saved['children'])


class StateTask:
    def __init__(self, record, root, device='cpu'):
        self.record, self.task = record, record['task']
        self.context = dict(record.get('context', {}))
        self.root = Path(root)
        for name in ('oracle',):
            if name in self.context:
                self.context[name] = str((self.root/self.context[name]).resolve())
        self.checkpoint = (self.root/record['map_checkpoint']).resolve()
        reader_context=dict(self.context)
        if self.task=='bridge':reader_context={'times':list(range(1,len(self.context['times'])+1))}
        self.reader = copy(frozen_reader(self.task, str(self.checkpoint),
            json.dumps(reader_context, sort_keys=True), device))
        self.env = getattr(self.reader, 'env', None)
        if self.task=='bridge':
            from .bridge.environment import Bridge
            self.env=Bridge(self.context['times'])
            self.reader.env=self.env
            self.reader._features=Bridge(self.env.times,feature_encoding=self.reader.config.get('input_encoding','identity')).features
        if self.task in ('cube2', 'puzzle8', 'path'):
            self.env = teacher_environment(self.task, self.context)
        if self.task == 'blocks':
            from .blocks.rules import solver
            self.solve = solver()
        self.tables = {}

    def state(self, value):
        if self.task in ('blocks','bridge','cube2','puzzle8','path'):
            return int(value)
        if self.task == 'game24':
            return self.env.canonical(value)
        return tuple(map(int, value))

    def transitions(self, state):
        if self.task == 'blocks':
            from .blocks import rules
            for tile, action in rules.PLACEMENTS:
                if state & tile == tile:
                    key = f"remove({action['shape_id']},{action['row']},{action['col']})"
                    yield key, state ^ tile, 1
        elif self.task in ('cube2','puzzle8','path'):
            for action, child in zip(self.env.actions,self.env.successors([state])[0]):
                if child >= 0:yield action, int(child), 1
        else:
            for action, child, cost in self.env.transitions(state):
                if self.task == 'bridge':action = list(self.env.actions[action])
                elif self.task == 'game24':action = list(self.env.actions[action])
                yield action_key(action), child, float(cost)

    def step(self, state, action):
        for key, child, _ in self.transitions(state):
            if key == action:return child
        raise ValueError('Illegal physical action')

    def distance(self, state, goal):
        if self.task == 'blocks':
            if state & goal != goal:return float('inf')
            cost, _ = self.solve(state ^ goal)
            return float('inf') if cost is None else cost
        if self.task == 'bridge':
            if goal not in self.tables:self.tables[goal] = self.env.graph(goal)['distances']
            return float(self.tables[goal][state])
        if self.task == 'token_swapping':
            from .token_swapping.relation_data import rank
            relative = self.env.relative_permutation(state, goal)
            table = np.memmap(self.context['oracle'],dtype=np.uint8,mode='r',shape=(3628800,))
            return int(table[int(rank(np.asarray(relative)))])
        if self.task=='game24':
            if state==goal:return 0
            if len(state)<=len(goal):return float('inf')
            key=(state,goal)
            if key not in self.tables:
                self.tables[key]=min((1+self.distance(child,goal) for _,child,_ in self.env.transitions(state)),default=float('inf'))
            return self.tables[key]
        if self.task=='rush_hour':
            index,oracle=rush_teacher(self.context['oracle'])
            if goal not in self.tables:self.tables[goal]=oracle.to_goal(index[goal])
            return float(self.tables[goal][index[state]])
        if self.task=='puzzle8' and goal not in self.env.goal_columns:
            if not hasattr(self.env,'extra_goal_tables'):self.env.extra_goal_tables={}
            if goal not in self.env.extra_goal_tables:
                original=self.env.tables,self.env.goal_columns
                self.env.prepare_goals([goal]);self.env.extra_goal_tables[goal]=self.env.tables[0].copy()
                self.env.tables,self.env.goal_columns=original
            return int(self.env.extra_goal_tables[goal][state])
        return float(self.env.distance(state, goal))

    def render(self, state):
        if self.task == 'blocks':
            from .blocks.rules import to_grid
            return to_grid(state).replace('/', '\n')
        if self.task == 'bridge':
            mask, bank = divmod(state,2)
            return json.dumps(dict(left=[p for p in range(self.env.n) if not (mask>>p)&1],
                right=[p for p in range(self.env.n) if (mask>>p)&1],torch='right' if bank else 'left'))
        if self.task == 'rush_hour':return '\n'.join(self.env.render(state))
        if self.task == 'game24':return str([str(x) for x in state])
        if self.task in ('cube2','puzzle8'):
            from .restoration_sft import render
            return render(self.task,self.env,state)
        return str(state)

    def make_step(self, state, goal):
        candidates = list(self.transitions(state))
        vectors = self.reader.encode([state,goal,*[c[1] for c in candidates]]).cpu()
        scores = self.reader.distance(vectors,vectors[1]).cpu().numpy()
        totals = np.asarray([c[2] for c in candidates])+scores[2:]
        optimal = np.flatnonzero(np.isclose(totals,totals.min())).tolist() if len(totals) else []
        return Step(state,goal,tuple(c[0] for c in candidates),tuple(c[1] for c in candidates),
            tuple(map(float,scores[2:])),float(scores[0]),tuple(i+1 for i in optimal),
            q_map_batch(vectors),self.render(state))


def prompt(task, env, step, executed, start, first, with_map=True):
    record = env.record if hasattr(env,'record') else env
    candidates='\n'.join(f'{i}: {a}' for i,a in enumerate(step.candidate_actions,1))
    update = ('[Environment update]\nCurrent state:\n'+step.current_text+
        '\nActual executed actions: '+str(executed or '(none)')+'\nLegal next moves:\n'+
        candidates+
        '\nCandidate numbers may change between turns; use only the numbers above.\n[/Environment update]')
    if not first:return update
    formatting = ('First output "Map-distance ranking to the goal, closest to farthest: RANKING." '
        'Rank the nearest 10 candidates plus current using < and =. ' if with_map else 'Do not output candidate rankings. ')
    formatting += 'Output a short factual Reason, then <action>ID</action>; at the exact goal output a short Reason and <done/>.'
    return record['initial_text']+'\n\n'+formatting+'\n\n'+update


def answer(step, chosen, with_map):
    if chosen is None:return 'Reason: The current state exactly matches the complete goal.\n<done/>'
    action, child = step.execute(chosen)
    text = ('Map-distance ranking to the goal, closest to farthest: '+_ranking(step,10)+'.\n') if with_map else ''
    text += 'Reason: '+action+' is legal and '+('reaches the complete goal.' if child==step.goal else 'preserves a minimum-cost completion.')
    return text+f'\n<action>{chosen}</action>'


def demonstration(record, steps, rng=None, with_map=True):
    turns=[];path=[steps[0].current];executed=[]
    for step,action in zip(steps,[*record['reference_actions'],None]):
        if rng is not None:step=numbered_step(step,tuple(map(int,rng.permutation(len(step.candidate_actions)))))
        chosen=step.candidate_actions.index(action)+1 if action is not None else None
        turns.append(SupervisedTurn(prompt(record['task'],record,step,executed,record['start'],not turns,with_map),
            answer(step,chosen,with_map),step,tuple(path),chosen))
        if chosen is not None:
            physical,state=step.execute(chosen);executed.append(physical);path.append(state)
    if path[-1]!=steps[-1].goal:raise ValueError('Incomplete certified demonstration')
    return Demonstration(tuple(turns),tuple(path),True)


def make_step(env, reader, state, goal):
    return env.make_step(state,goal)


def score(text, step, env):
    before=float(env.distance(step.current,step.goal));candidate,error=parse_control(text,len(step.candidate_actions))
    row=dict(current=str(step.current),goal=str(step.goal),answer=text,candidate=candidate,error=error,
        live=not step.done,rank_eligible=not step.done,safe=False,optimal=False,map_optimal=False)
    if candidate is not None and error is None:
        action,child=step.execute(candidate);remaining=env.distance(child,step.goal)
        cost=next(c for a,_,c in env.transitions(step.current) if a==action)
        row.update(after=str(child),safe=bool(np.isfinite(remaining)),optimal=bool(np.isclose(cost+remaining,before)),
            map_optimal=candidate in step.map_minimal_candidates)
    if not step.done:row.update(meeting_ranking_scores(step.candidate_map_distances,parse_ranking(text,len(step.candidate_actions),10)))
    return row
