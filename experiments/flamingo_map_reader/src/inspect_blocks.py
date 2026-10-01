"""Audit actual SFT lengths with the selected tokenizer before model loading."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from .blocks import FrozenBoardMap
from .blocks_data import demonstration_from_record
from .transcript import encode_trajectory


def main():
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Model ID or local tokenizer directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest["config"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    qmap = FrozenBoardMap.load(manifest["q_checkpoint"])
    lengths, action_lengths, over_limit, longest = [], [], [], []
    for index, record in enumerate(manifest["records"]):
        if record["split"] != "train":
            continue
        demo = demonstration_from_record(qmap, record, config["maximum_demonstration_actions"])
        encoded = encode_trajectory(demo, tokenizer, 1_000_000,
                                    chat_template_kwargs=config["chat_template_kwargs"])
        length = len(encoded.input_ids)
        lengths.append(length)
        longest.append((length, index))
        for turn in demo.turns:
            if not turn.step.done:
                action_lengths.append(len(tokenizer.encode(turn.answer_text, add_special_tokens=False)))
        if length > config["maximum_sequence_tokens"]:
            over_limit.append(index)
        if len(lengths) % 1000 == 0:
            print(json.dumps({"checked": len(lengths), "over_limit": len(over_limit)}), flush=True)
    result = {"samples": len(lengths), "sequence_limit": config["maximum_sequence_tokens"],
              "sequence_token_percentiles": dict(zip(("min", "median", "p90", "p99", "max"),
                      np.percentile(lengths, (0, 50, 90, 99, 100)).tolist())),
              "over_limit_record_indices": over_limit,
              "max_action_answer_tokens": max(action_lengths),
              "action_answers_over_generation_budget": sum(length > config["evaluation"]["action_max_new_tokens"]
                                                              for length in action_lengths),
              "longest_record_indices": [index for _, index in sorted(longest, reverse=True)[:64]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
