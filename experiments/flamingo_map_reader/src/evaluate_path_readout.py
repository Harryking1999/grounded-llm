"""Evaluate small and expanded readers on exactly the same held-out questions."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from .data import first_turn_from_record, load_graph
from .evaluate_graph_readout import generate_answer
from .prepare_path_readout import non_tied_pairs
from .readout_aux import paired_turns
from .short_readout import parse_short_answer
from .train import build_reader


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--record-stop", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest["config"]
    saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    trained = saved["contract"]
    if (trained["map_source"] != str(args.source_root.resolve()) or
            trained["model_source"] != str(args.model_path.resolve()) or
            Path(manifest["source_root"]).resolve() != args.source_root.resolve() or
            trained["config"]["task"] != "graph" or trained["config"]["map"] != config["map"]):
        raise ValueError("Held-out comparison must use the same model, maps and interface")
    if args.record_start < 0 or (args.record_stop is not None and args.record_stop <= args.record_start):
        raise ValueError("Invalid record range")
    if args.out.exists():
        raise FileExistsError(args.out)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    records = [r for r in manifest["records"] if r["evaluate"]]
    records = records[args.record_start:args.record_stop]
    graphs, counts = {}, defaultdict(Counter)
    args.out.mkdir(parents=True)
    with (args.out / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            graph = record["graph_id"]
            if graph not in graphs:
                graphs[graph] = load_graph(args.source_root, graph)[:2]
            turn = first_turn_from_record(*graphs[graph], record)
            pair_rows = []
            bucket = Counter(records=1)
            for pair in non_tied_pairs(turn.step):
                answers = []
                for example in paired_turns(turn, "pairwise", pair=pair):
                    answer = generate_answer(reader, tokenizer, example.step, example.user_text,
                        config, device, config["evaluation"]["action_max_new_tokens"],
                        stop_markers=("</closer>",))
                    chosen = parse_short_answer(answer, "closer", len(turn.step.candidate_actions))
                    answers.append(dict(answer=answer, chosen=chosen,
                        expected=example.chosen_id, correct=chosen == example.chosen_id))
                valid = all(a["chosen"] in pair for a in answers)
                correct = all(a["correct"] for a in answers)
                bucket.update(pairs=1, original_correct=int(answers[0]["correct"]),
                    swapped_correct=int(answers[1]["correct"]), paired_correct=int(correct),
                    both_legal=int(valid), choice_changed=int(valid and answers[0]["chosen"] != answers[1]["chosen"]))
                pair_rows.append(dict(pair=pair, original=answers[0], q_swap=answers[1]))
            bucket["all_pairs_correct"] = int(bucket["pairs"] > 0 and bucket["paired_correct"] == bucket["pairs"])
            split = record["split"]
            for group in (split, f"{split}:{graph}", f"{split}:length_{record['shortest_moves']}"):
                counts[group].update(bucket)
            handle.write(json.dumps(dict(record=record, comparisons=pair_rows, score=dict(bucket))) + "\n")
            handle.flush()
    summary = dict(adapter_checkpoint=str(args.adapter_checkpoint.resolve()),
        training_manifest=trained["manifest"], evaluation_manifest=str(args.manifest.resolve()),
        record_range=[args.record_start, args.record_stop], generation_mode="free",
        maximum_new_tokens=config["evaluation"]["action_max_new_tokens"],
        readout={key: dict(value) for key, value in counts.items()})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
