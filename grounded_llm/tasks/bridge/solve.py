"""Capacity-two all-left optimum, with the standard two-strategy recurrence."""


def optimal_plan(times):
    remaining = sorted(range(len(times)), key=lambda p: (times[p], p))
    actions = []
    while len(remaining) > 3:
        a, b, y, z = remaining[0], remaining[1], remaining[-2], remaining[-1]
        if times[a] + 2 * times[b] + times[z] <= 2 * times[a] + times[y] + times[z]:
            actions.extend([[a, b], [a], [y, z], [b]])
        else:
            actions.extend([[a, z], [a], [a, y], [a]])
        remaining = remaining[:-2]
    if len(remaining) == 3:
        a, b, c = remaining
        actions.extend([[a, b], [a], [a, c]])
    elif remaining:
        actions.append(remaining)
    return actions, sum(max(times[p] for p in action) for action in actions)
