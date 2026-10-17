"""Hanoi configuration rules with one-goal BFS, including the new eight-disk case."""
from collections import deque


def step(state, action):
    if not isinstance(action, dict) or set(action) != {"disk_id", "from", "to"}:
        raise ValueError("Expected disk_id/from/to object")
    disk, source, target = action["disk_id"], action["from"], action["to"]
    if type(disk) is not int or not 1 <= disk <= len(state):
        raise ValueError("Invalid disk_id")
    if not isinstance(source, str) or not isinstance(target, str) or source not in ("A", "B", "C") or target not in ("A", "B", "C") or source == target:
        raise ValueError("Invalid pegs")
    src, dst = ord(source) - 65, ord(target) - 65
    if state[disk - 1] != src or any(p == src for p in state[:disk - 1]):
        raise ValueError("Specified disk is not on top of source")
    if any(p == dst for p in state[:disk - 1]):
        raise ValueError("Larger disk on smaller disk")
    result = list(state)
    result[disk - 1] = dst
    return tuple(result)


def successors(state):
    top = [next((d for d, p in enumerate(state) if p == peg), len(state)) for peg in range(3)]
    for src in range(3):
        for dst in range(3):
            if src != dst and top[src] < top[dst]:
                action = {"disk_id": top[src] + 1, "from": chr(65 + src), "to": chr(65 + dst)}
                yield action, step(state, action)


def distances(goal):
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
    return "\n".join(chr(65 + peg) + ": " + str([d + 1 for d in reversed(range(len(state))) if state[d] == peg]) for peg in range(3))
