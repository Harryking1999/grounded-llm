"""Free-generation graph evaluation on the reserved five-graph suite."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_graph
from .evaluate_blocks import parse_control
from .graph import graph_step, timeline_maps
from .prompt import initial_prompt, turn_prompt
from .train import build_reader, to_device


def rollout(reader, tokenizer, environment, qmap, case, config, device):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class ControlBoundary(StoppingCriteria):
        def __init__(self, prefix_length):
            self.prefix_length = prefix_length

        def __call__(self, input_ids, scores, **kwargs):
            text = tokenizer.decode(input_ids[0, self.prefix_length:], skip_special_tokens=True)
            return "</action>" in text or "<done/>" in text

    start, goal = int(case["start"]), int(case["goal"])
    rng = np.random.default_rng(config["seed"] + int(case["id"].split("_")[-1]))
    first = initial_prompt(environment.adjacency, start, goal,
                           node_order=case["node_order"],
                           neighbor_order={int(k): v for k, v in case["neighbors"].items()})
    path = [start]
    messages, steps, trace = [], [], []
    previous, old_ids = [], []
    reached = start == goal
    failure = "action_budget"
    for turn in range(config["maximum_demonstration_actions"] + 1):
        step = graph_step(environment, qmap, path[-1], goal, executed_path=path, rng=rng)
        steps.append(step)
        update = turn_prompt(step, path)
        messages.append({"role": "user", "content": first + "\n\n" + update if turn == 0 else update})
        prefix = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, **config.get("chat_template_kwargs", {}))
        if prefix[:len(previous)] != previous:
            raise ValueError("Chat template changed historical tokens")
        ids = old_ids + [turn] * (len(prefix) - len(previous))
        if len(prefix) >= config["maximum_sequence_tokens"]:
            failure = "context_budget"
            break
        limit = config["evaluation"]["terminal_max_new_tokens" if step.done else "action_max_new_tokens"]
        limit = min(limit, config["maximum_sequence_tokens"] - len(prefix))
        timeline = to_device(timeline_maps(steps, ids), device)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with torch.inference_mode(), context:
            output = reader.generate(timeline, input_ids=torch.tensor([prefix], device=device),
                max_new_tokens=limit, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([ControlBoundary(len(prefix))]))
        answer = tokenizer.decode(output[0, len(prefix):], skip_special_tokens=True).strip()
        item = {"current": path[-1], "answer": answer,
                "candidate_destinations": step.candidate_destinations,
                "generated_tokens": output.shape[1] - len(prefix)}
        trace.append(item)
        try:
            chosen = parse_control(answer)
        except ValueError:
            failure = "format_error"
            break
        if chosen is None:
            failure = None if step.done else "false_done"
            break
        if step.done:
            failure = "action_after_goal"
            break
        if not step.candidate_actions:
            failure = "no_legal_actions"
            break
        if turn == config["maximum_demonstration_actions"]:
            failure = "action_budget"
            break
        try:
            _, destination = step.execute(environment, chosen)
        except ValueError:
            failure = "invalid_candidate"
            break
        path.append(destination)
        reached = reached or destination == goal
        item.update(chosen_id=chosen, after=destination)
        messages.append({"role": "assistant", "content": answer})
        complete = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=False, **config.get("chat_template_kwargs", {}))
        if complete[:len(prefix)] != prefix:
            raise ValueError("Chat template changed generation prefix")
        old_ids = ids + [turn] * (len(complete) - len(prefix))
        previous = complete
    return {"case_id": case["id"], "path": path, "trace": trace,
            "reached_goal": reached, "success": failure is None,
            "shortest_success": failure is None and len(path) - 1 == case["reference"]["length"],
            "failure": failure}


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config != manifest["config"] or args.source_root.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Evaluation source differs from the frozen graph contract")
    saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    expected = {"config": config, "manifest": str(args.manifest.resolve()),
                "map_source": str(args.source_root.resolve()),
                "model_source": str(args.model_path.resolve()) if args.model_path else config["model"]}
    if any(saved["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter was trained on another map contract")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    model_source = str(args.model_path) if args.model_path else config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    base = AutoModelForCausalLM.from_pretrained(model_source,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    cases = []
    graphs = {}
    for graph_id in [*config["train_validation_graphs"], config["unseen_test_graph"]]:
        environment, qmap, suite = load_graph(args.source_root, graph_id)
        graphs[graph_id] = environment, qmap
        cases.extend((graph_id, case) for case in suite["cases"])
    if args.limit is not None:
        cases = cases[:args.limit]
    (args.out / "logs").mkdir(parents=True)
    (args.out / "results").mkdir()
    counts = {}
    with (args.out / "logs/rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for graph_id, case in cases:
            result = rollout(reader, tokenizer, *graphs[graph_id], case, config, device)
            result["graph_id"] = graph_id
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            key = graph_id
            row = counts.setdefault(key, Counter())
            row.update(attempts=1, reached=int(result["reached_goal"]),
                success=int(result["success"]), shortest=int(result["shortest_success"]))
            if result["failure"]:
                row[result["failure"]] += 1
    summary = {key: dict(row, goal_reach_rate=row["reached"] / row["attempts"],
                         shortest_success_rate=row["shortest"] / row["attempts"])
               for key, row in counts.items()}
    (args.out / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
