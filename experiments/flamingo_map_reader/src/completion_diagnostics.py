"""Matched full-completion readout: original, Q swap, renumbering and token CE."""

from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .convergence import with_decision_mask
from .data import first_turn_from_record, load_graph
from .diagnose_interface import DecisionPositionRecorder, decision_position
from .evaluate_graph_readout import generate_answer
from .prepare_path_readout import non_tied_pairs
from .readout_aux import paired_turns, renumber_turn
from .relations import first_ranked_candidate, parse_ranking, score_relationships
from .sft import Demonstration
from .train import MapCollator, to_device
from .transcript import encode_trajectory


def diagnostic_records(records, config, final=False):
    """Use the same stratified fixed cases at every epoch and for both losses."""
    eligible = [r for r in records if r["evaluate"]]
    if final:
        return eligible
    selected = []
    for split_index, split in enumerate(("train", "validation", "test")):
        groups = defaultdict(list)
        for record in eligible:
            if record["split"] == split:
                groups[(record["graph_id"], record["shortest_moves"])].append(record)
        rng = np.random.default_rng(config["evaluation"]["diagnostic_seed"] + split_index)
        keys = sorted(groups)
        rng.shuffle(keys)
        for rows in groups.values():
            rng.shuffle(rows)
        budget = config["evaluation"]["diagnostic_records_per_split"]
        while keys and budget:
            for key in keys[:]:
                selected.append(groups[key].pop())
                budget -= 1
                if not groups[key]:
                    keys.remove(key)
                if not budget:
                    break
    return selected


def ordered_pair_correct(step, answer, pair):
    ranks = parse_ranking(answer, len(step.candidate_actions))
    if ranks is None:
        return False
    left, right = pair
    return ((ranks[left] < ranks[right]) ==
            (step.candidate_map_distances[left - 1] < step.candidate_map_distances[right - 1])
            and ranks[left] != ranks[right])


def generated_score(reader, tokenizer, turn, config, device):
    answer = generate_answer(reader, tokenizer, turn.step, turn.user_text, config,
                             device, config["evaluation"]["action_max_new_tokens"])
    score = score_relationships(turn.step, answer)
    first = first_ranked_candidate(answer, len(turn.step.candidate_actions))
    score["first_candidate_correct"] = first in turn.step.map_minimal_candidates
    score["legal_action"] = (score["chosen_id"] is not None and
                             1 <= score["chosen_id"] <= len(turn.step.candidate_actions))
    return {"answer": answer, "score": score}


@torch.inference_mode()
def completion_loss(reader, tokenizer, collator, turn, config, device, interface=False):
    encoded = encode_trajectory(Demonstration((turn,), turn.executed_path, True), tokenizer,
                                config["maximum_sequence_tokens"],
                                chat_template_kwargs=config.get("chat_template_kwargs", {}))
    example = with_decision_mask((encoded, [turn.step]), tokenizer)
    position = decision_position(example)
    inputs = collator([example])
    maps = to_device(inputs.pop("map_batch"), device)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    focus = inputs.pop("focus_mask")[:, 1:]
    labels = inputs["labels"][:, 1:]
    selected = focus & (labels != -100)
    recorder = DecisionPositionRecorder(reader, position) if interface else None
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with precision, recorder if recorder else nullcontext():
        outputs = reader(maps, **inputs)
    ranking_logits = outputs.logits[:, :-1][selected].float()
    total_tokens = int((labels != -100).sum())
    result = {"total_nll": float(outputs.loss) * total_tokens,
              "total_tokens": total_tokens,
              "ranking_nll": float(F.cross_entropy(ranking_logits, labels[selected], reduction="sum")),
              "ranking_tokens": int(selected.sum()),
              "first_ranking_token_correct": int(outputs.logits[0, position].argmax() == labels[0, position]),
              "examples": 1}
    return result, recorder


def interface_comparison(original, swapped, pair):
    rows = []
    left, right = pair
    for base, other, hidden, changed in zip(original.adapters, swapped.adapters,
                                           original.residuals, swapped.residuals):
        rows.append({"layer": base["layer"], "gate": base["gate"],
            "delta_over_hidden": base["delta_over_hidden"],
            "swap_delta_over_hidden": math.dist(hidden, changed) /
                (math.sqrt(sum(value * value for value in hidden)) + 1e-12),
            "attention_left": base["attention"][left + 1],
            "attention_right": base["attention"][right + 1],
            "swapped_attention_left": other["attention"][left + 1],
            "swapped_attention_right": other["attention"][right + 1],
            "reproduction_error": base["reproduction_error"]})
    return rows


def evaluate_completion(reader, tokenizer, config, records, source_root, output, *,
                        shard_index=0, shard_count=1, final=False):
    """Every score uses free generation; loss probes use separate gold forwards."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = next(reader.parameters()).device
    collator = MapCollator(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    graphs, scores, losses = {}, defaultdict(Counter), defaultdict(Counter)
    interface_seen = Counter()
    chosen = diagnostic_records(records, config, final)[shard_index::shard_count]
    with (output / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for record in chosen:
            graph = record["graph_id"]
            if graph not in graphs:
                graphs[graph] = load_graph(source_root, graph)[:2]
            turn = first_turn_from_record(*graphs[graph], record)
            # The source first turn may choose a tied minimum randomly; all gold
            # completions use the same canonical legal minimum as paired training.
            pairs = non_tied_pairs(turn.step)
            turn = paired_turns(turn, "full_ranking", pair=pairs[0])[0]
            split = record["split"]
            groups = (split, f"{split}:{graph}", f"{split}:length_{record['shortest_moves']}")
            inspect = interface_seen[split] < config["evaluation"]["interface_records_per_split"]
            original = generated_score(reader, tokenizer, turn, config, device)
            loss, original_hidden = completion_loss(reader, tokenizer, collator, turn, config, device, inspect)
            for group in groups:
                losses[f"{group}:original"].update(loss)
            base_score = original["score"]
            count = Counter(records=1, original_exact=int(base_score["exact_ranking"]),
                original_first_correct=int(base_score["first_candidate_correct"]),
                original_action_correct=int(base_score["action_map_minimum"]),
                original_legal=int(base_score["legal_action"]))
            variants = []
            for pair_index, pair in enumerate(pairs):
                swapped_turn = paired_turns(turn, "full_ranking", pair=pair)[1]
                swapped = generated_score(reader, tokenizer, swapped_turn, config, device)
                loss, swapped_hidden = completion_loss(reader, tokenizer, collator, swapped_turn,
                                                       config, device, inspect and pair_index == 0)
                for group in groups:
                    losses[f"{group}:swap"].update(loss)
                swap_score = swapped["score"]
                count.update(pairs=1,
                    paired_order_correct=int(ordered_pair_correct(turn.step, original["answer"], pair)
                        and ordered_pair_correct(swapped_turn.step, swapped["answer"], pair)),
                    paired_exact=int(base_score["exact_ranking"] and swap_score["exact_ranking"]),
                    paired_first_correct=int(base_score["first_candidate_correct"] and swap_score["first_candidate_correct"]),
                    paired_action_correct=int(base_score["action_map_minimum"] and swap_score["action_map_minimum"]),
                    both_legal=int(base_score["legal_action"] and swap_score["legal_action"]))
                variant = {"pair": pair, **swapped}
                if swapped_hidden:
                    variant["interface"] = interface_comparison(original_hidden, swapped_hidden, pair)
                variants.append(variant)
            count["all_pairs_correct"] = int(count["paired_order_correct"] == len(pairs))
            renumbered = renumber_turn(turn, "graph", list(reversed(range(len(turn.step.candidate_actions)))))
            renamed = generated_score(reader, tokenizer, renumbered, config, device)
            loss, _ = completion_loss(reader, tokenizer, collator, renumbered, config, device)
            for group in groups:
                losses[f"{group}:renumber"].update(loss)
            same_action = False
            if base_score["legal_action"] and renamed["score"]["legal_action"]:
                same_action = (turn.step.candidate_actions[base_score["chosen_id"] - 1] ==
                               renumbered.step.candidate_actions[renamed["score"]["chosen_id"] - 1])
            count.update(renumber_same_physical_action=int(same_action),
                         renumber_exact=int(renamed["score"]["exact_ranking"]),
                         renumber_action_correct=int(renamed["score"]["action_map_minimum"]))
            for group in groups:
                scores[group].update(count)
            handle.write(json.dumps({"record": record, "original": original,
                "swaps": variants, "renumber": renamed, "score": dict(count)}) + "\n")
            handle.flush()
            interface_seen[split] += 1
    summary = {"records": len(chosen), "final": final, "shard_index": shard_index,
               "shard_count": shard_count, "scores": dict(scores), "loss_sums": dict(losses)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
