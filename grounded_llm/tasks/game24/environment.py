"""XoT-style pair reduction; exact rationals replace float sentinel arithmetic."""
from fractions import Fraction
from itertools import combinations
import math
import numpy as np
from grounded_llm.core.qmap.exact_graph import goal_distances


class Game24:
    actions = tuple((i, op, j) for i, j in combinations(range(4), 2)
                    for op in ('+', '-', 'r-', '*', '/', 'r/'))
    goal = (Fraction(24),)

    def __init__(self, max_numbers=4):
        if max_numbers < 2: raise ValueError('At least two input slots required')
        self.max_numbers=max_numbers
        if max_numbers != 4:
            self.actions=tuple((i,op,j) for i,j in combinations(range(max_numbers),2)
                               for op in ('+','-','r-','*','/','r/'))

    def validate(self, state):
        if not 1 <= len(state) <= self.max_numbers or tuple(state)!=self.canonical(state):
            raise ValueError('Expected a canonical exact rational multiset within slot capacity')

    @staticmethod
    def canonical(numbers):
        return tuple(sorted(Fraction(x) for x in numbers))

    def transitions(self, state):
        for action, (i, op, j) in enumerate(self.actions):
            if j >= len(state): continue
            x, y = state[i], state[j]
            if (op == '/' and y == 0) or (op == 'r/' and x == 0): continue
            if op == '+': z = x + y
            elif op == '-': z = x - y
            elif op == 'r-': z = y - x
            elif op == '*': z = x * y
            elif op == '/': z = x / y
            else: z = y / x
            yield action, tuple(sorted([z] + [v for k, v in enumerate(state) if k not in (i, j)])), 1

    def features(self, states):
        out = np.zeros((len(states), self.max_numbers, 4), dtype=np.float32)
        for row, state in enumerate(states):
            for col, value in enumerate(state):
                out[row, col] = (1, math.copysign(math.log1p(abs(value.numerator)), value.numerator),
                                 math.log1p(value.denominator), float(value) / (1 + abs(float(value))))
        return out.reshape(len(states), -1)

    def graph(self, numbers):
        start = self.canonical(numbers)
        self.validate(start)
        states, index = [start], {start: 0}
        rows = []
        for state in states:
            row = [-1] * len(self.actions)
            for a, child, _ in self.transitions(state):
                if child not in index:
                    index[child] = len(states)
                    states.append(child)
                row[a] = index[child]
            rows.append(row)
        children = np.asarray(rows, dtype=np.int32)
        costs = np.ones_like(children, dtype=np.float32)
        terminal = np.asarray([s == self.goal for s in states])
        distances = goal_distances(children, costs, terminal)
        return dict(features=self.features(states), children=children, costs=costs,
                    baseline=np.asarray([min(abs(float(v) - 24) for v in s) for s in states], dtype=np.float32),
                    terminal=terminal, distances=distances,
                    goal_features=self.features([self.goal])[0], starts=np.array([0]),
                    metadata={"numbers": list(numbers), "state_count": len(states),
                              "exact_rational_states": [[str(v) for v in s] for s in states]})

    def judge(self, numbers, actions, goal=None):
        goal=self.goal if goal is None else self.canonical(goal)
        if not 1 <= len(goal) <= len(numbers) <= self.max_numbers or len(actions)!=len(numbers)-len(goal):
            return {"success": False, "reason": "operand_or_action_count_mismatch"}
        state = self.canonical(numbers)
        for action in actions:
            choices = {self.actions[a]: child for a, child, _ in self.transitions(state)}
            if tuple(action) not in choices:
                return {"success": False, "reason": "illegal_action"}
            state = choices[tuple(action)]
        return {"success": state == goal, "result": str(state[0]) if len(state)==1 else [str(v) for v in state]}
