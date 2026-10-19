"""Full-trajectory targets and consistent action renumbering, independent of training."""
from grounded_llm.core.reader.memory import MapBatch
from grounded_llm.core.reader.numbering import numbered_step

from dataclasses import replace
from itertools import permutations

import numpy as np

from .sft import Demonstration, decision_text




def numbering_plans(demo, count, seed):
    """Balanced per-turn permutations and distinct whole-trajectory variants.

    For k <= 3 cycle all k! permutations. Larger catalogues sample distinct
    permutations. The physical demonstration and its tie choices stay fixed.
    """
    rng = np.random.default_rng(seed)
    per_turn = []
    for turn in demo.turns:
        k = len(turn.step.candidate_actions) if not turn.step.done else 0
        if k <= 3:
            pool = list(permutations(range(k)))
            rng.shuffle(pool)
        else:
            pool = []
            while len(pool) < count:
                order = tuple(map(int, rng.permutation(k)))
                if order not in pool:
                    pool.append(order)
        per_turn.append([pool[i % len(pool)] for i in range(count)])
    plans = []
    for i in range(count):
        plan = tuple(orders[i] for orders in per_turn)
        if plan not in plans:
            plans.append(plan)
    # k=1 throughout legitimately admits only one physical numbering.
    return plans


def renumber_demonstration(demo, plan, task, reported=None):
    from .prompt import turn_prompt as graph_prompt
    from grounded_llm.tasks.blocks.prompts import turn_prompt as blocks_prompt
    if len(plan) != len(demo.turns):
        raise ValueError("A numbering plan must cover every turn")
    turns, actions = [], []
    for turn, order in zip(demo.turns, plan):
        # Terminal maps can still contain legal actions, but have no decision.
        step = turn.step if turn.step.done else numbered_step(turn.step, order)
        prompt = graph_prompt(step, turn.executed_path) if task == "graph" else blocks_prompt(step, actions)
        if not turns:
            prompt = turn.user_text.split("[Environment update]", 1)[0] + prompt
        chosen = None
        answer = turn.answer_text
        if not step.done and turn.chosen_id is not None:
            actual = turn.step.candidate_actions[turn.chosen_id - 1]
            chosen = step.candidate_actions.index(actual) + 1
            answer = decision_text(step, chosen, reported)
            actions.append(actual)
        turns.append(replace(turn, step=step, chosen_id=chosen,
                             user_text=prompt, answer_text=answer))
    return Demonstration(tuple(turns), demo.executed_path, demo.success, demo.no_solution)
