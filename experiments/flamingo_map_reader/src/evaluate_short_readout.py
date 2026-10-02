"""Short pairwise/nearest map readout, separate from full-ranking task scores."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .blocks import FrozenBoardMap
from .blocks_data import demonstration_from_record
from .counterfactual import swap_best_worst_q
from .data import demonstration_from_record as graph_demonstration, load_graph
from .evaluate_graph_readout import generate_answer
from .short_readout import correct_ids, diagnostic_prompt, parse_short_answer
from .train import build_reader


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("pairwise", "nearest"), required=True)
    parser.add_argument("--scaffold-prefix", action="store_true")
    parser.add_argument("--max-train-turns", type=int, default=16)
    parser.add_argument("--max-validation-turns", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    task = config["task"]
    source = args.q_checkpoint if task == "blocks" else args.source_root
    source_field = "q_checkpoint" if task == "blocks" else "source_root"
    if (config != manifest["config"] or task not in ("graph", "blocks") or
            source is None or source.resolve() != Path(manifest[source_field]).resolve()):
        raise ValueError("Short readout needs its frozen task pilot contract")
    saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    expected = {"config": config, "manifest": str(args.manifest.resolve()),
                "map_source": str(source.resolve()),
                "model_source": str(args.model_path.resolve())}
    if any(saved["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter was trained with another map contract")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    qmap = FrozenBoardMap.load(source) if task == "blocks" else None
    graphs = {}
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    counts = defaultdict(Counter)
    with (args.out / "first_turns.jsonl").open("w", encoding="utf-8") as handle:
        for record in manifest["records"]:
            split = record["split"]
            maximum = args.max_train_turns if split == "train" else args.max_validation_turns
            if split not in ("train", "validation") or counts[split]["cases"] >= maximum:
                continue
            if task == "blocks":
                turn = demonstration_from_record(qmap, record,
                    config["maximum_demonstration_actions"]).turns[0]
            else:
                graph_id = record["graph_id"]
                if graph_id not in graphs:
                    graphs[graph_id] = load_graph(source, graph_id)[:2]
                turn = graph_demonstration(*graphs[graph_id], record).turns[0]
            step = turn.step
            swapped = swap_best_worst_q(step)
            if swapped is None:
                continue
            distances = step.candidate_map_distances
            pair = (step.map_minimal_candidates[0], int(np.argmax(distances)) + 1)
            prompt, tag = diagnostic_prompt(turn.user_text, args.mode, pair)
            prefix = f"<{tag}>" if args.scaffold_prefix else ""
            answers = []
            for condition in (step, swapped):
                answer = generate_answer(reader, tokenizer, condition, prompt, config,
                    device, 64, assistant_prefix=prefix,
                    stop_markers=(f"</{tag}>",))
                chosen = parse_short_answer(answer, tag, len(step.candidate_actions))
                answers.append({"answer": answer, "chosen_id": chosen,
                                "correct": chosen in correct_ids(condition, args.mode, pair)})
            both_valid = all(item["chosen_id"] is not None for item in answers)
            counts[split].update(cases=1, base_valid=int(answers[0]["chosen_id"] is not None),
                swapped_valid=int(answers[1]["chosen_id"] is not None),
                base_correct=int(answers[0]["correct"]),
                swapped_correct=int(answers[1]["correct"]),
                paired_correct=int(all(item["correct"] for item in answers)),
                choice_changed=int(both_valid and
                                   answers[0]["chosen_id"] != answers[1]["chosen_id"]))
            handle.write(json.dumps({"split": split, "record": record,
                "pair": pair, "original": answers[0], "q_swap": answers[1]}) + "\n")
            handle.flush()
    summary = {"task": task, "mode": args.mode,
               "generation_mode": "scaffold_diagnostic" if args.scaffold_prefix else "free",
               "readout": {split: dict(bucket) for split, bucket in counts.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
