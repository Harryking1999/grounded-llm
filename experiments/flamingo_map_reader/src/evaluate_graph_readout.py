"""Fixed first-turn Q-distance readout, separate from closed-loop planning."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch

from .data import demonstration_from_record, load_graph
from .graph import graph_step, timeline_maps
from .prompt import initial_prompt, turn_prompt
from .relations import score_relationships
from .summarize_graph_eval import add_relation, relation_summary
from .text import chat_ids
from .train import build_reader, to_device


def fixed_turns(config, manifest, source_root, splits=("validation", "reserved")):
    graphs = {}
    for graph_id in [*config["train_validation_graphs"], config["unseen_test_graph"]]:
        graphs[graph_id] = load_graph(source_root, graph_id)
    for index, record in enumerate(manifest["records"]):
        if record["split"] not in splits or record["start"] == record["goal"]:
            continue
        environment, qmap, _ = graphs[record["graph_id"]]
        turn = demonstration_from_record(environment, qmap, record).turns[0]
        yield record["split"], record["graph_id"], f"manifest:{index}", turn.step, turn.user_text
    for graph_id in ([*config["train_validation_graphs"], config["unseen_test_graph"]]
                     if "reserved" in splits else []):
        environment, qmap, suite = graphs[graph_id]
        for case in suite["cases"]:
            if case["start"] == case["goal"]:
                continue
            rng = np.random.default_rng(config["seed"] + int(case["id"].split("_")[-1]))
            step = graph_step(environment, qmap, int(case["start"]), int(case["goal"]),
                              executed_path=[int(case["start"])], rng=rng)
            opening = initial_prompt(environment.adjacency, int(case["start"]), int(case["goal"]),
                node_order=case["node_order"],
                neighbor_order={int(k): v for k, v in case["neighbors"].items()})
            yield "reserved", graph_id, case["id"], step, opening + "\n\n" + turn_prompt(step, [int(case["start"])])


def generate_answer(reader, tokenizer, step, user_text, config, device, maximum_new_tokens,
                    assistant_prefix="", stop_markers=("</action>", "<done/>")):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class ControlBoundary(StoppingCriteria):
        def __init__(self, prefix_length):
            self.prefix_length = prefix_length

        def __call__(self, input_ids, scores, **kwargs):
            answer = tokenizer.decode(input_ids[0, self.prefix_length:], skip_special_tokens=True)
            return any(marker in answer for marker in stop_markers)

    prefix = chat_ids(tokenizer, [{"role": "user", "content": user_text}],
                      add_generation_prompt=True, **config.get("chat_template_kwargs", {}))
    if assistant_prefix:
        prefix += tokenizer.encode(assistant_prefix, add_special_tokens=False)
    if len(prefix) >= config["maximum_sequence_tokens"]:
        raise ValueError("Fixed first-turn prompt exceeds context")
    timeline = to_device(timeline_maps([step], [0] * len(prefix)), device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), context:
        output = reader.generate(timeline, input_ids=torch.tensor([prefix], device=device),
            attention_mask=torch.ones((1, len(prefix)), dtype=torch.long, device=device),
            max_new_tokens=min(maximum_new_tokens, config["evaluation"]["action_max_new_tokens"],
                               config["maximum_sequence_tokens"] - len(prefix)),
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([ControlBoundary(len(prefix))]))
    return assistant_prefix + tokenizer.decode(output[0, len(prefix):], skip_special_tokens=True).strip()


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "reserved"),
                        default=("validation", "reserved"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    gate = json.loads(args.gate_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config != manifest["config"] or args.source_root.resolve() != Path(manifest["source_root"]).resolve():
        raise ValueError("Readout source differs from the frozen graph contract")
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

    args.out.mkdir(parents=True)
    counts = defaultdict(Counter)
    observed = Counter()
    with (args.out / "first_turns.jsonl").open("w", encoding="utf-8") as handle:
        for split, graph_id, case_id, step, user_text in fixed_turns(
                config, manifest, args.source_root, args.splits):
            answer = generate_answer(reader, tokenizer, step, user_text, config, device,
                                     gate["maximum_new_tokens"])
            score = score_relationships(step, answer)
            observed[split] += 1
            for group in (split, f"{split}:{graph_id}"):
                add_relation(counts[group], score)
            handle.write(json.dumps({"split": split, "graph_id": graph_id,
                "case_id": case_id, "answer": answer, "score": score}) + "\n")
            handle.flush()
    for split in ("validation", "reserved"):
        if split in args.splits and observed[split] != gate[f"{split}_first_turns"]:
            raise ValueError(f"Unexpected {split} fixed-turn coverage: {dict(observed)}")
    summary = relation_summary(counts)
    required = {"valid_ranking_rate": gate["minimum_valid_ranking_rate"],
                "pairwise_accuracy": gate["minimum_pairwise_accuracy"],
                "closest_candidate_accuracy": gate["minimum_closest_candidate_accuracy"]}
    summary["readout_gate"] = {
        split: {"passed": all(summary[split][metric] >= threshold
                              for metric, threshold in required.items()),
                "thresholds": required}
        for split in ("validation", "reserved") if split in args.splits
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: summary[split] for split in (*args.splits, "readout_gate")}))


if __name__ == "__main__":
    main()
