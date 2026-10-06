"""Closed-loop blocks evaluation with per-turn maps and environment judging."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import re

import numpy as np
import torch

from .blocks import FrozenBoardMap, blocks_step
from .blocks_data import coverage_group
from .blocks_prompt import initial_prompt, turn_prompt
from .graph import timeline_maps
from .train import build_reader, to_device
from .text import chat_ids


def parse_control(answer):
    markers = re.findall(r"<action>\s*([0-9]+|none)\s*</action>|(<done/>)", answer)
    if markers == [("none", ""), ("", "<done/>")]:
        return None
    if len(markers) != 1:
        raise ValueError("Expected exactly one action or done marker")
    action, done = markers[0]
    if action == "none":
        raise ValueError("Action none requires a following done marker")
    return None if done else int(action)


def rollout(reader, tokenizer, qmap, record, config, device):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class ControlBoundary(StoppingCriteria):
        def __init__(self, prefix_length):
            self.prefix_length = prefix_length

        def __call__(self, input_ids, scores, **kwargs):
            text = tokenizer.decode(input_ids[0, self.prefix_length:], skip_special_tokens=True)
            return "</action>" in text or "<done/>" in text

    rng = np.random.default_rng(record["sample_seed"])
    current, goal = int(record["start"]), int(record["goal"])
    initial = initial_prompt(current, goal)
    messages, steps, trace, actions = [], [], [], []
    previous, old_ids = [], []
    reached = current == goal
    kwargs = config["chat_template_kwargs"]
    for turn in range(config["maximum_demonstration_actions"] + 1):
        step = blocks_step(qmap, current, goal, rng=rng)
        steps.append(step)
        user = turn_prompt(step, actions)
        messages.append({"role": "user", "content": initial + "\n\n" + user if turn == 0 else user})
        prefix = chat_ids(tokenizer, messages, add_generation_prompt=True, **kwargs)
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
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=device),
                max_new_tokens=limit, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([ControlBoundary(len(prefix))]))
        answer = tokenizer.decode(output[0, len(prefix):], skip_special_tokens=True).strip()
        item = {"current": str(current), "answer": answer,
                "candidate_actions": list(step.candidate_actions),
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
            action, actual = step.execute(chosen)
        except ValueError:
            failure = "invalid_candidate"
            break
        actions.append(action)
        current = actual
        reached = reached or current == goal
        item.update(chosen_id=chosen, executed_action=action, after=str(current))
        messages.append({"role": "assistant", "content": answer})
        complete = chat_ids(tokenizer, messages, add_generation_prompt=False, **kwargs)
        if complete[:len(prefix)] != prefix:
            raise ValueError("Chat template changed generation prefix")
        old_ids = ids + [turn] * (len(complete) - len(prefix))
        previous = complete
    else:
        raise AssertionError("rollout must terminate within budget")
    return {"record": record, "reached_goal": reached, "success": failure is None,
            "failure": failure, "executed_actions": actions, "trace": trace}


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config != manifest["config"]:
        raise ValueError("Manifest and evaluation configurations differ")
    # Load adapter-only model weights; Trainer's RNG/optimizer files are separate.
    saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    expected = {"config": config, "manifest": str(args.manifest.resolve()),
                "map_source": str(args.q_checkpoint.resolve()),
                "model_source": str(args.model_path.resolve()) if args.model_path else config["model"]}
    if any(saved["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter was trained with a different contract")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    model_source = str(args.model_path) if args.model_path else config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    base = AutoModelForCausalLM.from_pretrained(model_source,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    qmap = FrozenBoardMap.load(args.q_checkpoint)
    seen = set(map(int, manifest["training_visible_states"]))
    summaries = {}
    (args.out / "logs").mkdir(parents=True)
    (args.out / "results").mkdir()
    with (args.out / "logs/rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for record in manifest["records"]:
            if record["split"] != "test":
                continue
            group = coverage_group(int(record["start"]), int(record["goal"]), seen)
            if group != record["group"]:
                raise ValueError("Test coverage group disagrees with SFT states")
            result = rollout(reader, tokenizer, qmap, record, config, device)
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            counts = summaries.setdefault(group, Counter())
            counts.update(attempts=1, reached_goal=int(result["reached_goal"]), success=int(result["success"]))
            if result["failure"]:
                counts[result["failure"]] += 1
    summary = {group: dict(counts, goal_reach_rate=counts["reached_goal"] / counts["attempts"],
                          success_rate=counts["success"] / counts["attempts"])
               for group, counts in summaries.items()}
    (args.out / "results/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
