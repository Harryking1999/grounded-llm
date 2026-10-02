"""Paired first-turn ranking, Q swap, and candidate renumbering diagnostics."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .blocks import FrozenBoardMap
from .blocks_data import demonstration_from_record as blocks_demonstration
from .blocks_prompt import turn_prompt as blocks_turn_prompt
from .counterfactual import swap_best_worst_q
from .data import demonstration_from_record as graph_demonstration, load_graph
from .evaluate_graph_readout import generate_answer
from .memory import MapBatch
from .prompt import turn_prompt as graph_turn_prompt
from .relations import first_ranked_candidate, score_relationships
from .summarize_graph_eval import add_relation, relation_summary
from .train import build_reader


def reorder_candidates(step, order):
    """Keep physical actions and Q vectors paired while assigning new local IDs."""
    count = len(step.candidate_actions)
    if sorted(order) != list(range(count)):
        raise ValueError("candidate order must be a permutation")
    slots = [0, 1, *(index + 2 for index in order)]
    source = step.map_batch
    ids = torch.tensor([[0, 0, *range(1, count + 1)]], dtype=torch.long,
                       device=source.candidate_ids.device)
    distances = tuple(step.candidate_map_distances[index] for index in order)
    minimum = min(distances)
    best = tuple(index for index, distance in enumerate(distances, 1)
                 if np.isclose(distance, minimum, rtol=1e-10, atol=1e-12))
    return replace(step,
        map_batch=MapBatch(source.vectors[:, slots], source.roles[:, slots], ids,
                           source.valid[:, slots]),
        candidate_actions=tuple(step.candidate_actions[index] for index in order),
        candidate_destinations=tuple(step.candidate_destinations[index] for index in order),
        candidate_map_distances=distances, map_minimal_candidates=best)


def renumbered_prompt(task, step, original_text):
    marker = "\n\n[Environment update]"
    if marker not in original_text:
        raise ValueError("first-turn prompt lacks its environment update")
    opening = original_text.rsplit(marker, 1)[0]
    update = (graph_turn_prompt(step, [step.current]) if task == "graph" else
              blocks_turn_prompt(step, []))
    return opening + "\n\n" + update


def physical_action(step, chosen_id):
    return (step.candidate_actions[chosen_id - 1]
            if chosen_id is not None and 1 <= chosen_id <= len(step.candidate_actions)
            else None)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--q-checkpoint", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--perturbation-limit", type=int, default=4)
    parser.add_argument("--scaffold-prefix", action="store_true",
                        help="Diagnostic only: supply a common answer opening")
    parser.add_argument("--max-train-turns", type=int, default=0)
    parser.add_argument("--max-validation-turns", type=int, default=0)
    parser.add_argument("--maximum-new-tokens", type=int,
                        help="Diagnostic cap; report separately from the formal task budget")
    args = parser.parse_args()
    if args.maximum_new_tokens is not None and args.maximum_new_tokens <= 0:
        raise ValueError("maximum new tokens must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config != manifest["config"]:
        raise ValueError("Pilot manifest/config mismatch")
    task = config["task"]
    source = args.source_root if task == "graph" else args.q_checkpoint
    field = "source_root" if task == "graph" else "q_checkpoint"
    if source is None or source.resolve() != Path(manifest[field]).resolve():
        raise ValueError("Frozen map source differs from pilot manifest")
    saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    expected = {"config": config, "manifest": str(args.manifest.resolve()),
                "map_source": str(source.resolve()),
                "model_source": str(args.model_path.resolve())}
    if any(saved["contract"].get(key) != value for key, value in expected.items()):
        raise ValueError("Adapter was trained with another pilot contract")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    reader = build_reader(base, config).to(device).eval()
    reader.load_adapter_state_dict(saved["adapter"])
    qmap = FrozenBoardMap.load(source) if task == "blocks" else None
    graph_cache = {}
    counts = defaultdict(Counter)
    perturbed = Counter()
    perturb_counts = Counter()
    args.out.mkdir(parents=True)
    assistant_prefix = ("The current state has not reached the goal.\n"
                        "Map-distance ranking to the goal, closest to farthest: "
                        if args.scaffold_prefix else "")
    maximum_new_tokens = (args.maximum_new_tokens or
                          config["evaluation"]["action_max_new_tokens"])
    with (args.out / "first_turns.jsonl").open("w", encoding="utf-8") as handle:
        for record in manifest["records"]:
            split = record["split"]
            if split not in ("train", "validation"):
                continue
            limit = args.max_train_turns if split == "train" else args.max_validation_turns
            if limit and counts[split]["turns"] >= limit:
                continue
            if task == "graph":
                graph_id = record["graph_id"]
                if graph_id not in graph_cache:
                    graph_cache[graph_id] = load_graph(source, graph_id)[:2]
                demo = graph_demonstration(*graph_cache[graph_id], record)
            else:
                demo = blocks_demonstration(qmap, record,
                    config["maximum_demonstration_actions"])
            turn = demo.turns[0]
            step, user_text = turn.step, turn.user_text
            if step.done or not step.candidate_actions:
                continue
            answer = generate_answer(reader, tokenizer, step, user_text, config,
                                     device, maximum_new_tokens,
                                     assistant_prefix=assistant_prefix)
            score = score_relationships(step, answer)
            add_relation(counts[split], score)
            first = first_ranked_candidate(answer, len(step.candidate_actions))
            counts[split].update(first_ranked_candidate_present=int(first is not None),
                first_ranked_candidate_correct=int(first in step.map_minimal_candidates))
            row = {"split": split, "record": record, "answer": answer,
                   "score": score}
            if split == "train" and perturbed[split] < args.perturbation_limit and len(step.candidate_actions) > 1:
                swapped = swap_best_worst_q(step)
                if swapped is not None:
                    swapped_answer = generate_answer(reader, tokenizer, swapped,
                        user_text, config, device,
                        maximum_new_tokens,
                        assistant_prefix=assistant_prefix)
                    swap_score = score_relationships(swapped, swapped_answer)
                    swapped_first = first_ranked_candidate(swapped_answer,
                        len(swapped.candidate_actions))
                    base_first = first_ranked_candidate(answer, len(step.candidate_actions))
                    base_chosen = score["chosen_id"]
                    swap_chosen = swap_score["chosen_id"]
                    both_valid = (base_chosen is not None and swap_chosen is not None and
                                  1 <= base_chosen <= len(step.candidate_actions) and
                                  1 <= swap_chosen <= len(step.candidate_actions))
                    perturb_counts.update(q_swap_cases=1,
                        q_swap_both_valid=int(both_valid),
                        q_swap_action_changed=int(both_valid and base_chosen != swap_chosen),
                        q_swap_valid_ranking=int(swap_score["valid_ranking"]),
                        q_swap_exact_ranking=int(swap_score["exact_ranking"]),
                        q_swap_paired_exact=int(score["exact_ranking"] and
                                                swap_score["exact_ranking"]),
                        q_swap_new_minimum=int(swap_score["action_map_minimum"]),
                        q_swap_first_correct=int(swapped_first in swapped.map_minimal_candidates),
                        q_swap_paired_first_correct=int(base_first in step.map_minimal_candidates and
                                                        swapped_first in swapped.map_minimal_candidates),
                        q_swap_first_changed=int(base_first is not None and
                                                 swapped_first is not None and
                                                 base_first != swapped_first),
                        q_swap_pairwise_correct=swap_score["pairwise_correct"],
                        q_swap_pairwise_total=swap_score["pairwise_total"])
                    row["q_swap"] = {"answer": swapped_answer, "score": swap_score}
                reordered = reorder_candidates(step,
                    list(reversed(range(len(step.candidate_actions)))))
                reordered_answer = generate_answer(reader, tokenizer, reordered,
                    renumbered_prompt(task, reordered, user_text), config, device,
                    maximum_new_tokens,
                    assistant_prefix=assistant_prefix)
                reordered_score = score_relationships(reordered, reordered_answer)
                base_actual = physical_action(step, score["chosen_id"])
                reordered_actual = physical_action(reordered, reordered_score["chosen_id"])
                both_valid = base_actual is not None and reordered_actual is not None
                perturb_counts.update(renumber_cases=1,
                    renumber_both_valid=int(both_valid),
                    renumber_same_physical_action=int(both_valid and base_actual == reordered_actual),
                    renumber_new_minimum=int(reordered_score["action_map_minimum"]))
                row["renumber"] = {"answer": reordered_answer,
                                   "score": reordered_score,
                                   "same_physical_action": both_valid and base_actual == reordered_actual}
                perturbed[split] += 1
            handle.write(json.dumps(row) + "\n")
            handle.flush()
    summary = {"readout": relation_summary(counts),
               "perturbations": dict(perturb_counts),
               "requested_perturbation_limit": args.perturbation_limit,
               "generation_max_new_tokens": maximum_new_tokens,
               "generation_mode": "scaffold_diagnostic" if args.scaffold_prefix else "free"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
