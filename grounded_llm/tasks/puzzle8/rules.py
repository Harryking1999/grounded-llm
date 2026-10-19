"""8-puzzle rules and exact single-goal BFS; no model-side search."""
from collections import deque

MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


def validate(state):
    if len(state) != 9 or any(type(x) is not int for x in state) or sorted(state) != list(range(9)):
        raise ValueError("Expected a permutation of 0..8")


def step(state, action):
    if not isinstance(action, str) or action not in MOVES:
        raise ValueError("Unknown direction")
    row, col = divmod(state.index(0), 3)
    dr, dc = MOVES[action]
    nr, nc = row + dr, col + dc
    if not (0 <= nr < 3 and 0 <= nc < 3):
        raise ValueError("Blank leaves board")
    result = list(state)
    a, b = row * 3 + col, nr * 3 + nc
    result[a], result[b] = result[b], result[a]
    return tuple(result)


def successors(state):
    for action in MOVES:
        try:
            yield action, step(state, action)
        except ValueError:
            continue


def distances(goal):
    validate(goal)
    goal = tuple(goal)
    result, queue = {goal: 0}, deque([goal])
    while queue:
        state = queue.popleft()
        for _, child in successors(state):
            if child not in result:
                result[child] = result[state] + 1
                queue.append(child)
    return result


def solve(state, table):
    state, actions = tuple(state), []
    while table[state]:
        action, child = next((a, c) for a, c in successors(state) if table[c] == table[state] - 1)
        actions.append(action)
        state = child
    return actions


def render(state):
    return "\n".join(" ".join(map(str, state[i:i + 3])) for i in (0, 3, 6))
