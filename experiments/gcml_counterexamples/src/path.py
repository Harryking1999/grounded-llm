"""Exact execution and shortest paths in (node, switch-mask) product graphs."""
from collections import deque


def transition(case, node, mask, destination):
    if type(destination) is not int or destination not in case["neighbors"][str(node)]:
        raise ValueError("nonexistent graph edge")
    edge = sorted([node, destination])
    for gate in case.get("gates", []):
        if edge == gate["edge"] and not (mask & (1 << gate["bit"])):
            raise ValueError(f"closed gate {gate['bit']}")
    for switch in case.get("switches", []):
        if destination == switch["node"]:
            mask ^= 1 << switch["bit"]
    return destination, mask


def shortest(case):
    start = (case["start"], case.get("initial_mask", 0))
    queue, parents = deque([start]), {start: None}
    end = None
    while queue:
        state = queue.popleft()
        if state[0] == case["goal"]:
            end = state
            break
        for node in case["neighbors"][str(state[0])]:
            try:
                after = transition(case, *state, node)
            except ValueError:
                continue
            if after not in parents:
                parents[after] = state
                queue.append(after)
    if end is None:
        return None
    states = []
    while end is not None:
        states.append(end)
        end = parents[end]
    states.reverse()
    return {"length": len(states) - 1,
            "states": [{"node": node, "mask": mask} for node, mask in states],
            "path": [{"from": a[0], "to": b[0]} for a, b in zip(states, states[1:])]}


def judge(case, output):
    path = output.get("path") if isinstance(output, dict) else None
    if not isinstance(path, list):
        return {"pass": False, "execution_pass": False, "failure_type": "missing_path", "trace": []}
    reference = shortest(case)
    if reference is None:
        raise ValueError("Unsolvable evaluator case")
    node, mask = case["start"], case.get("initial_mask", 0)
    trace, illegal = [], None
    limit = case.get("step_limit")
    for step, move in enumerate(path if limit is None else path[:limit], 1):
        try:
            if not isinstance(move, dict) or type(move.get("from")) is not int or move["from"] != node:
                raise ValueError("from does not match current node")
            node, mask = transition(case, node, mask, move.get("to"))
        except ValueError as error:
            illegal = {"step": step, "reason": str(error), "move": move}
            break
        reported = move.get("switches_after")
        expected = [(mask >> bit) & 1 for bit in range(case.get("bits", 0))]
        trace.append({"step": step, "node": node, "mask": mask,
                      "reported_switches_after": reported,
                      "switch_report_correct": reported == expected if case.get("bits") else None})
    execution = illegal is None and node == case["goal"] and (limit is None or len(path) <= limit)
    optimal = execution and len(path) == reference["length"]
    passed = optimal if case.get("objective") == "shortest" else execution
    failure = ("illegal_move" if illegal else "too_many_moves" if limit is not None and len(path) > limit
               else "goal_not_reached" if node != case["goal"]
               else "non_shortest" if not passed else None)
    return {"pass": passed, "execution_pass": execution, "optimal": optimal,
            "shortest_moves": reference["length"], "attempted_moves": len(path),
            "executed_moves": len(trace), "excess_moves": len(path) - reference["length"] if execution else None,
            "final_node_actual": node, "final_mask_actual": mask,
            "failure_type": failure, "illegal_move": illegal,
            "switch_reports_correct": sum(t["switch_report_correct"] is True for t in trace),
            "switch_reports_evaluated": len(trace) if case.get("bits") else 0, "trace": trace}
