"""Shared free-generation evaluator: every reference turn and actual closed-loop rollouts."""

from argparse import ArgumentParser
from collections import defaultdict, deque
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from .blocks import FrozenBoardMap, blocks_step
from . import blocks_prompt, prompt, sft
from .data import load_graph, shortest_move_counts
from .evaluate_blocks import parse_control
from .graph import graph_step, timeline_maps
from .text import chat_ids
from .train import build_reader, to_device
from .trajectory_dataset import load_record
from .trajectory_metrics import score_turn, summarize_turns


class GenerationSession:
    """Historical maps stay attached to historical text, including cached decoding."""
    def __init__(self, reader, tokenizer, config, device, no_map=False):
        self.reader, self.tokenizer, self.config, self.device = reader, tokenizer, config, device
        self.no_map = no_map
        self.messages, self.steps, self.previous, self.old_ids = [], [], [], []

    def ask(self, user, step):
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokenizer = self.tokenizer
        self.steps.append(step)
        self.messages.append(dict(role="user", content=user))
        prefix = chat_ids(tokenizer, self.messages, add_generation_prompt=True,
                          **self.config.get("chat_template_kwargs", {}))
        if prefix[:len(self.previous)] != self.previous:
            raise ValueError("Chat template changed the completed historical prefix")
        self.prefix = prefix
        self.ids = self.old_ids + [len(self.steps) - 1] * (len(prefix) - len(self.previous))
        remaining = self.config["maximum_sequence_tokens"] - len(prefix)
        if remaining <= 0:
            return "", dict(generated_tokens=0, seconds=0., context_exhausted=True)
        limit = min(remaining, self.config["evaluation"]["terminal_max_new_tokens" if step.done else "action_max_new_tokens"])

        class Boundary(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, len(prefix):], skip_special_tokens=True)
                return "</action>" in text or "<done/>" in text

        inputs = dict(input_ids=torch.tensor([prefix], device=self.device),
            attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=self.device),
            max_new_tokens=limit, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([Boundary()]))
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        started = time.monotonic()
        with torch.inference_mode(), context:
            if self.no_map:
                output = self.reader.base_model.generate(**inputs)
            else:
                output = self.reader.generate(to_device(timeline_maps(self.steps, self.ids), self.device), **inputs)
        text = tokenizer.decode(output[0, len(prefix):], skip_special_tokens=True).strip()
        return text, dict(generated_tokens=output.shape[1] - len(prefix), seconds=time.monotonic() - started,
                          context_exhausted=False)

    def accept(self, answer):
        self.messages.append(dict(role="assistant", content=answer))
        complete = chat_ids(self.tokenizer, self.messages, add_generation_prompt=False,
                            **self.config.get("chat_template_kwargs", {}))
        if complete[:len(self.prefix)] != self.prefix:
            raise ValueError("Chat template changed this turn's generation prefix")
        self.old_ids = self.ids + [len(self.steps) - 1] * (len(complete) - len(self.prefix))
        self.previous = complete


class TaskEnvironment:
    """Only executes and judges: its distances are never supplied as model input."""
    def __init__(self, config, manifest):
        self.config = config
        self.task = config["task"]
        if self.task == "graph":
            self.env, self.qmap, _ = load_graph(Path(manifest["source_root"]), config["data"]["graph_id"])
            self.distances = shortest_move_counts(self.env.adjacency)
        else:
            from experiments.blocks_distance_map.src.oracle import DistanceOracle
            self.qmap = FrozenBoardMap.load(Path(manifest["q_checkpoint"]))
            self.oracle = DistanceOracle(cache_limit=1_000_000, seconds=1_000_000)

    def step(self, path, goal, rng):
        if self.task == "graph":
            return graph_step(self.env, self.qmap, path[-1], goal, executed_path=path, rng=rng)
        return blocks_step(self.qmap, path[-1], goal, rng=rng)

    def execute(self, step, chosen):
        return step.execute(self.env, chosen) if self.task == "graph" else step.execute(chosen)

    def remaining(self, current, goal, path=None):
        if self.task != "graph":
            return self.oracle.distance(current, goal)
        if not path:
            return int(self.distances[current, goal])
        visited = set(path[:-1])
        queue = deque([(current, 0)])
        visited.add(current)
        while queue:
            node, distance = queue.popleft()
            if node == goal:
                return distance
            for nxt in np.flatnonzero(self.env.adjacency[node]):
                if int(nxt) not in visited:
                    visited.add(int(nxt))
                    queue.append((int(nxt), distance + 1))
        return -1

    def chosen_is_shortest(self, step, chosen, path, remaining):
        if chosen is None or not 1 <= chosen <= len(step.candidate_actions) or remaining < 1:
            return False
        destination = step.candidate_destinations[chosen - 1]
        return self.remaining(destination, step.goal, [*path, destination]) == remaining - 1

    def chosen_keeps_reachable(self, step, chosen, path, remaining):
        """The task's own criterion: the goal is still reachable after this action.

        Wider than chosen_is_shortest. On blocks one figure is built from several
        pieces, so removing any of its components is as good as removing the one
        the map ranks first -- there is no unique shortest path among the good
        moves. remaining() answers -1 for an unreachable state on both tasks.
        """
        if chosen is None or not 1 <= chosen <= len(step.candidate_actions) or remaining < 1:
            return False
        destination = step.candidate_destinations[chosen - 1]
        return self.remaining(destination, step.goal, [*path, destination]) >= 0

    def reachable_candidates(self, step, path, remaining):
        """How many legal candidates leave the goal reachable: the chance floor.

        A model picking uniformly among the legal moves scores this fraction on
        action_keeps_goal_reachable, so the rate is only readable beside it.
        """
        if remaining < 1:
            return 0
        return sum(self.remaining(destination, step.goal, [*path, destination]) >= 0
                   for destination in step.candidate_destinations)

    def update(self, step, path, actions):
        return prompt.turn_prompt(step, path) if self.task == "graph" else blocks_prompt.turn_prompt(step, actions)

    def terminal(self, path, actions):
        return sft.terminal_text(path) if self.task == "graph" else blocks_prompt.terminal_text(actions)


def reference_turns(session, demo, environment):
    rows = []
    for turn_index, turn in enumerate(demo.turns):
        answer, usage = session.ask(turn.user_text, turn.step)
        row = dict(turn=turn_index, remaining_shortest=environment.remaining(turn.step.current, turn.step.goal, turn.executed_path),
            **score_turn(turn.step, answer, turn.answer_text if turn.step.done else None,
                         environment.config["data"].get("reported_candidates")), **usage,
            answer=answer, current=str(turn.step.current), goal=str(turn.step.goal))
        chosen = row.get("chosen_id")
        row["chosen_action"] = turn.step.candidate_actions[chosen - 1] if row["legal_action"] else None
        row["action_environment_shortest"] = environment.chosen_is_shortest(turn.step, chosen, turn.executed_path, row["remaining_shortest"])
        row["action_keeps_goal_reachable"] = environment.chosen_keeps_reachable(turn.step, chosen, turn.executed_path, row["remaining_shortest"])
        row["reachable_candidates"] = environment.reachable_candidates(turn.step, turn.executed_path, row["remaining_shortest"])
        row["candidate_slots"] = 0 if turn.step.done else len(turn.step.candidate_actions)
        rows.append(row)
        # Only earlier gold turns enter the next prefix. Never prepend this
        # turn's gold distances/ranking before generating its decision.
        session.accept(turn.answer_text)
    return rows


def closed_loop(session, record, first, environment, config, variant):
    start, goal = int(record["start"]), int(record["goal"])
    path, actions, rows, failure = [start], [], [], "action_budget"
    rng = np.random.default_rng(record["sample_seed"] + 104729 * (variant + 1))
    for index in range(config["maximum_demonstration_actions"] + 1):
        step = environment.step(path, goal, rng)
        user = environment.update(step, path, actions)
        if index == 0:
            user = first + user
        answer, usage = session.ask(user, step)
        row = dict(turn=index, remaining_shortest=environment.remaining(step.current, goal, path),
            **score_turn(step, answer, environment.terminal(path, actions) if step.done else None,
                         config["data"].get("reported_candidates")),
            **usage, answer=answer, current=str(step.current), goal=str(goal))
        rows.append(row)
        if usage["context_exhausted"]:
            failure = "context_budget"
            break
        try:
            chosen = parse_control(answer)
        except ValueError:
            failure = "format_error"
            break
        if chosen is None:
            failure = None if step.done else "premature_done"
            break
        if step.done:
            failure = "failed_to_stop"
            break
        if not row["legal_action"]:
            failure = "illegal_action"
            break
        if index == config["maximum_demonstration_actions"]:
            break
        action, destination = environment.execute(step, chosen)
        row["action_environment_shortest"] = environment.chosen_is_shortest(step, chosen, path, row["remaining_shortest"])
        row["action_keeps_goal_reachable"] = environment.chosen_keeps_reachable(step, chosen, path, row["remaining_shortest"])
        row["reachable_candidates"] = environment.reachable_candidates(step, path, row["remaining_shortest"])
        row["candidate_slots"] = len(step.candidate_actions)
        row["chosen_action"] = action
        path.append(destination)
        actions.append(action)
        session.accept(answer)
    return dict(turns=rows, reached_goal=goal in path, success=failure is None, failure=failure,
                moves=len(actions), shortest_moves=record["shortest_moves"],
                shortest_success=failure is None and len(actions) == record["shortest_moves"],
                actual_path=list(map(str, path)), actual_actions=actions)


def aggregate(results, manifest):
    rows = [dict(r, group=case["group"]) for case in results for r in case["turns"]]
    strata = {}
    for dimension in ("turn", "remaining_shortest", "candidates", "group"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        strata[dimension] = {key: summarize_turns(value) for key, value in groups.items()}
    physical = defaultdict(list)
    for case in results:
        physical[case["trajectory_id"]].append(case)
    consistent = 0
    comparable = 0
    for cases in physical.values():
        if len(cases) < 2 or cases[0]["mode"] != "reference":
            continue
        for turn in range(len(cases[0]["turns"])):
            selected = [c["turns"][turn].get("chosen_action") for c in cases]
            if cases[0]["turns"][turn]["done"]:
                continue
            comparable += 1
            consistent += None not in selected and len(set(selected)) == 1
    first = {key: cases[0]["turns"][0] for key, cases in physical.items() if cases[0]["turns"]}
    pairs = [(first[a], first[b]) for a, b in manifest["natural_target_pairs"] if a in first and b in first]
    units = defaultdict(list)
    record_by_id = {r["trajectory_id"]: r for r in manifest["records"]}
    for key, cases in physical.items():
        decision_rows = [r for c in cases for r in c["turns"] if not r["done"]]
        if decision_rows:
            record = record_by_id[key]
            unit = record.get("board_row", key)
            units[unit].append(sum(r.get("action_map_minimum", False) for r in decision_rows) / len(decision_rows))
    values = np.array([np.mean(v) for v in units.values()])
    ci = None
    if len(values) > 1:
        rng = np.random.default_rng(1729)
        means = values[rng.integers(len(values), size=(1000, len(values)))].mean(axis=1)
        ci = np.quantile(means, [.025, .975]).tolist()
    return dict(physical_tasks=len(physical), case_variants=len(results), overall=summarize_turns(rows), strata=strata,
        action_macro=dict(unit="initial_board" if manifest["config"]["task"] == "blocks" else "physical_task",
                          units=len(values), mean=float(values.mean()) if len(values) else None, bootstrap_95=ci),
        renumbering=dict(comparable_turns=comparable, same_physical_action=consistent),
        natural_goal_change=dict(pairs=len(pairs), both_correct=sum(a.get("action_map_minimum", False) and
            b.get("action_map_minimum", False) for a, b in pairs)),
        rollout=dict(attempts=sum(c["mode"] == "rollout" for c in results),
            reached=sum(c.get("reached_goal", False) for c in results), success=sum(c.get("success", False) for c in results),
            shortest_success=sum(c.get("shortest_success", False) for c in results),
            successful_moves=sum(c.get("moves", 0) for c in results if c.get("success")),
            successful_shortest_moves=sum(c.get("shortest_moves", 0) for c in results if c.get("success"))))


def main():
    parser = ArgumentParser()
    for arg in ("manifest", "adapter-checkpoint", "model-path", "out"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--mode", choices=("reference", "rollout"), required=True)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--no-map", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    config = manifest["config"]
    if args.out.exists():
        raise FileExistsError(args.out)
    saved = torch.load(args.adapter_checkpoint, weights_only=True, map_location="cpu")
    expected = dict(config=config, manifest=str(args.manifest.resolve()), model_source=str(args.model_path.resolve()),
                    map_source=manifest["source_root" if config["task"] == "graph" else "q_checkpoint"])
    if any(saved["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter and evaluation contracts differ")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    environment = TaskEnvironment(config, manifest)
    records = [r for r in manifest["records"] if r["split"] == args.split]
    if args.limit:
        # Deterministic spread across all physical records, including terminal-only examples.
        records = [records[i] for i in np.linspace(0, len(records) - 1, min(args.limit, len(records)), dtype=int)]
    records = records[args.start:args.stop]
    args.out.mkdir(parents=True)
    results = []
    with (args.out / "cases.jsonl").open("w") as handle:
        for record in records:
            for variant in range(min(args.variants, record["variants"])):
                demo = load_record(args.manifest.parent / "trajectories", record, config, variant)
                session = GenerationSession(reader, tokenizer, config, device, args.no_map)
                if args.mode == "reference":
                    value = dict(turns=reference_turns(session, demo, environment))
                else:
                    first = demo.turns[0].user_text.split("[Environment update]", 1)[0] if demo.turns else (
                        prompt.initial_prompt(environment.env.adjacency, int(record["start"]), int(record["goal"]))
                        if config["task"] == "graph" else blocks_prompt.initial_prompt(int(record["start"]), int(record["goal"])))
                    value = closed_loop(session, record, first, environment, config, variant)
                result = dict(value, trajectory_id=record["trajectory_id"], group=record["group"],
                    variant=variant, mode=args.mode, no_map=args.no_map, greedy_success=record["greedy_success"])
                results.append(result)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                print(json.dumps(dict(completed=len(results), trajectory_id=record["trajectory_id"], variant=variant)), flush=True)
    summary = aggregate(results, manifest)
    summary["contract"] = dict(checkpoint=str(args.adapter_checkpoint), split=args.split, mode=args.mode, no_map=args.no_map,
                               start=args.start, stop=args.stop, limit=args.limit, variants=args.variants)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
