"""Cube2 and 8-Puzzle text, map steps and trajectory augmentation."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch

from grounded_llm.core.reader.memory import MapBatch, q_map_batch
from grounded_llm.core.reader.numbering import numbered_step
from grounded_llm.core.sft.dataset import PreparedRoutes
from grounded_llm.core.sft.targets import _ranking
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.core.sft.types import Demonstration, SupervisedTurn
from grounded_llm.tasks.text import section_text
from grounded_llm.tasks.blocks.protocol import parse_control
from grounded_llm.core.sft.ranking import parse_ranking, meeting_ranking_scores


def environment(task, oracle_path=None):
    if task == "cube2":
        from .cube2.environment import Cube2
        return Cube2(oracle_path, metric="QTM_URF_6")
    if task == "puzzle8":
        from .puzzle8.environment import Puzzle8
        env = Puzzle8()
        env.prepare_goals([0])
        return env
    raise ValueError(task)


def load_map(task, env, checkpoint, device="cpu"):
    if task == "cube2":
        from .cube2.model import MapReader
    elif task == "puzzle8":
        from .puzzle8.model import MapReader
    else:
        raise ValueError(task)
    return MapReader(env, checkpoint, device)


@dataclass(frozen=True)
class Step:
    current: int
    goal: int
    candidate_actions: tuple
    candidate_destinations: tuple
    candidate_map_distances: tuple
    current_map_distance: float
    map_minimal_candidates: tuple
    map_batch: MapBatch

    @property
    def done(self):
        return self.current == self.goal

    def execute(self, candidate):
        if not 1 <= candidate <= len(self.candidate_actions):
            raise ValueError("Candidate outside legal list")
        return self.candidate_actions[candidate - 1], self.candidate_destinations[candidate - 1]


def make_step(env, reader, state, goal=0, vectors=None, distances=None):
    children = env.successors([state])[0]
    legal = np.flatnonzero(children >= 0)
    destinations = children[legal]
    ids = [state, goal, *map(int, destinations)]
    q = reader.encode(ids).cpu() if vectors is None else torch.from_numpy(np.array(vectors[ids]))
    d = (reader.distance(q, q[1]).numpy() if distances is None
         else np.asarray(distances[ids]))
    minimum = min(d[2:])
    return Step(int(state), int(goal), tuple(env.actions[int(i)] for i in legal),
                tuple(map(int, destinations)), tuple(map(float, d[2:])), float(d[0]),
                tuple(int(i + 1) for i in np.flatnonzero(d[2:] == minimum)), q_map_batch(q))


@lru_cache(None)
def templates(task):
    return (section_text(task, "共享任务描述与规则"),
            section_text(task, "地图组／CoT 组：每轮环境更新"))


def render(task, env, state):
    if task == "cube2":
        from .cube2.text import render as cube_render
        return cube_render(state)
    from .puzzle8.rules import render as board_render
    return board_render(tuple(map(int, env.positions[state])))


def prompt(task, env, step, executed, start, first):
    initial, update = templates(task)
    if task == "cube2":
        initial = initial.replace("{initial_cube_state}", render(task, env, start)).replace(
            "{goal_cube_state}", render(task, env, step.goal))
        update = update.replace("{current_cube_state}", render(task, env, step.current))
        candidate_template = "{local_id}: {turn}\n..."
    else:
        initial = initial.replace("{initial_board_3_rows}", render(task, env, start)).replace(
            "{goal_board_3_rows}", render(task, env, step.goal))
        update = update.replace("{current_board_3_rows}", render(task, env, step.current))
        candidate_template = "{local_id}: {direction}\n..."
    update = update.replace("{executed_actions_or_none}", " -> ".join(executed) or "(none)")
    update = update.replace(candidate_template, "\n".join(
        f"{i}: {action}" for i, action in enumerate(step.candidate_actions, 1)))
    if not first:
        return update
    return initial + "\n\n" + (
        'First output "Map-distance ranking to the goal, closest to farthest: RANKING." '
        'Rank the nearest 10 candidates plus current using < and =. '
        'Next output a short factual line beginning with "Reason:". '
        'Finally output <action>ID</action>. At the goal, output only a short '
        'Reason and <done/>.\n\n') + update


def answer(task, step, chosen):
    cube = task == "cube2"
    noun, move = ("cube state", "turn") if cube else ("board", "move")
    if chosen is None:
        return f"Reason: The current {noun} exactly matches the goal; no further {move} is needed.\n<done/>"
    action, destination = step.execute(chosen)
    verb = f"Turning {action}" if cube else f"Moving the blank {action.lower()}"
    ending = (f"reaches the goal {noun}" if destination == step.goal else
              f"leaves a {noun} with one fewer {move} needed to reach the goal")
    return (f"Map-distance ranking to the goal, closest to farthest: {_ranking(step, 10)}.\n"
            f"Reason: {verb} is legal and {ending}.\n<action>{chosen}</action>")


def demonstration(task, env, record, steps, rng=None):
    turns, path, executed = [], [record["start"]], []
    for original, action in zip(steps, [*record["reference_actions"], None]):
        step = (numbered_step(original, tuple(map(int, rng.permutation(len(original.candidate_actions)))))
                if rng is not None else original)
        chosen = step.candidate_actions.index(action) + 1 if action is not None else None
        turns.append(SupervisedTurn(prompt(task, env, step, executed, record["start"], not turns),
                                    answer(task, step, chosen), step, tuple(path), chosen))
        if chosen is not None:
            physical, state = step.execute(chosen)
            executed.append(physical)
            path.append(state)
    if path[-1] != record["goal"]:
        raise ValueError("Incomplete restoration trajectory")
    return Demonstration(tuple(turns), tuple(path), True)


class Routes(PreparedRoutes):
    def __init__(self, path, real_map, *, config, baseline, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        self.task = config["task"]
        self.env = environment(self.task)
        renumber = config["baselines"][baseline].get("renumber_each_epoch", False)
        super().__init__(path, real_map, transform=self.renumber_item if renumber else None)

    def renumber_item(self, row, item, epoch, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], epoch, index]))
        demo = demonstration(self.task, self.env, row, item["steps"], rng)
        encoded = encode_trajectory(demo, self.tokenizer, self.config["maximum_sequence_tokens"])
        return encoded, [turn.step for turn in demo.turns]


def score(text, step, env):
    before = int(env.distance(step.current, step.goal))
    candidate, error = parse_control(text, len(step.candidate_actions))
    row = dict(current=str(step.current), goal=str(step.goal), answer=text, candidate=candidate,
               error=error, live=before > 0, rank_eligible=not step.done,
               safe=False, optimal=False, map_optimal=False)
    if candidate is not None and error is None:
        _, after = step.execute(candidate)
        remaining = int(env.distance(after, step.goal))
        row.update(after=str(after), safe=before > 0 and remaining >= 0,
                   optimal=before > 0 and remaining == before - 1,
                   map_optimal=candidate in step.map_minimal_candidates)
    if not step.done:
        row.update(meeting_ranking_scores(step.candidate_map_distances,
                                         parse_ranking(text, len(step.candidate_actions), 10)))
    return row
