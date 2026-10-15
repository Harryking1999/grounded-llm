"""CLI for exact-reference or frozen-Q solving, with replayable actions."""
import argparse
import json
from pathlib import Path

from .tasks import make_environment


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--task", choices=("hanoi3", "hanoi4", "hanoi5", "hanoi6", "cube2", "cube2_qtm"), required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--goal", type=int)
    parser.add_argument("--scramble", default="", help="Cube U/R/F notation, space separated")
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Omit for exact reference solution")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    env = make_environment(args.task, args.oracle)
    source = 0 if args.start is None else args.start
    goal = (env.size - 1 if args.task.startswith("hanoi") else 0) if args.goal is None else args.goal
    env.validate(source)
    env.validate(goal)
    for action in args.scramble.split():
        source = env.step(source, action)
    if args.checkpoint is None:
        if env.name.startswith("cube2") and env.depth is None:
            if args.oracle is None:
                parser.error("Exact cube solving requires --oracle (builds it if missing)")
            env.build_oracle(args.oracle)
        actions = env.solve(source, goal)
        method, reason = "exact_bfs", "goal"
    else:
        from .model import MapReader
        reader = MapReader(env, args.checkpoint, args.device)
        state, seen, actions, reason = source, {source}, [], "budget"
        for _ in range(2 * env.diameter):
            if state == goal:
                reason = "goal"
                break
            action, state = reader.choose(state, goal)
            actions.append(action)
            if state in seen:
                reason = "cycle"
                break
            seen.add(state)
        method = "q_greedy"
    state, trace = source, [source]
    for action in actions:
        state = env.step(state, action)
        trace.append(state)
    print(json.dumps({"method": method, "start": env.describe(source), "goal": env.describe(goal),
                      "solved": state == goal, "stop": "goal" if state == goal else reason,
                      "actions": actions, "state_ids": trace}, indent=2))


if __name__ == "__main__":
    main()
