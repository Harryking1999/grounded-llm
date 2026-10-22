"""Path task text and demonstrations for the common SFT reader."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from grounded_llm.tasks.path.environment import GraphEnvironment, shortest_distances
from grounded_llm.core.reader.numbering import numbered_step
from grounded_llm.core.sft.targets import _ranking
from grounded_llm.core.sft.types import Demonstration, SupervisedTurn
from grounded_llm.tasks.text import section_text


@lru_cache(None)
def templates():
    return (section_text('path', '共享任务描述与规则'),
            section_text('path', '地图组／CoT 组：每轮环境更新'))


class Environment:
    """Existing fixed-graph rules plus exact labels for demonstrations and judging."""
    def __init__(self, graph):
        self.graph = GraphEnvironment.load(Path(graph))
        self.adjacency = self.graph.adjacency
        self.size = len(self.adjacency)
        self.actions = tuple(f'move({a},{b})' for a, b in self.graph.actions)
        self.next = np.full((self.size, len(self.actions)), -1, dtype=np.int32)
        for state in range(self.size):
            for action in self.graph.legal_actions(state):
                self.next[state, action] = self.graph.execute(state, action)
        self.distances = shortest_distances(self.adjacency)

    def successors(self, states):
        return self.next[np.asarray(states)]

    def step(self, state, action):
        index = self.actions.index(action)
        return self.graph.execute(state, index)

    def distance(self, source, goal):
        return self.distances[np.asarray(source), np.asarray(goal)]


def prompt(env, step, path, start, first, with_map=True):
    initial, update = templates()
    text = (update.replace('{local_id}: move({current_node},{next_node})\n...', '\n'.join(
                f'{i}: move({step.current},{child})'
                for i, child in enumerate(step.candidate_destinations, 1)))
            .replace('{current_node}', str(step.current))
            .replace('{visited_nodes_in_order}', ', '.join(map(str, path))))
    if not first:
        return text
    adjacency = '\n'.join(f'{node}: ' + ', '.join(map(str, env.adjacency[node].nonzero()[0]))
                          for node in range(len(env.adjacency)))
    initial = (initial.replace('{max_node_id}', str(len(env.adjacency)-1))
               .replace('{adjacency_list}', adjacency)
               .replace('{initial_node}', str(start))
               .replace('{goal_node}', str(step.goal)))
    formatting = ('First output "Map-distance ranking to the goal, closest to farthest: RANKING." '
                  'Rank the nearest 10 candidates plus current using < and =. '
                  if with_map else 'Do not output candidate rankings. ')
    formatting += ('Output a short factual line beginning with "Reason:", then <action>ID</action>. '
                   'At the goal, output a short Reason and <done/>.')
    return initial+'\n\n'+formatting+'\n\n'+text


def demonstration(env, record, steps, rng=None, with_map=True):
    path, turns = [record['start']], []
    for original, action in zip(steps, [*record['reference_actions'], None]):
        step = (numbered_step(original, tuple(map(int, rng.permutation(len(original.candidate_actions)))))
                if rng is not None else original)
        chosen = step.candidate_actions.index(action)+1 if action is not None else None
        text = prompt(env, step, path, record['start'], not turns, with_map)
        if chosen is None:
            answer = 'Reason: The current node is the goal; no further move is needed.\n<done/>'
        else:
            child = step.candidate_destinations[chosen-1]
            ending = ('reaches the goal node' if child == step.goal else
                      'leaves a route to the goal with one fewer move needed')
            ranking = (f'Map-distance ranking to the goal, closest to farthest: {_ranking(step,10)}.\n'
                       if with_map else '')
            answer = ranking+f'Reason: Moving from {step.current} to {child} is legal and {ending}.\n<action>{chosen}</action>'
        turns.append(SupervisedTurn(text, answer, step, tuple(path), chosen))
        if chosen is not None:
            assert env.step(step.current, action) == child
            path.append(child)
    if path[-1] != record['goal']:
        raise ValueError('Incomplete path trajectory')
    return Demonstration(tuple(turns), tuple(path), True)
