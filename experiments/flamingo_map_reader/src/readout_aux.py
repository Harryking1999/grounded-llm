"""Paired first-turn Q interventions for separate interface-readout training."""

from dataclasses import replace

import numpy as np

from torch.utils.data import Dataset

from .blocks_data import demonstration_from_record as blocks_demonstration
from .counterfactual import (swap_best_worst_q, swap_candidate_q,
                             reorder_candidates, renumbered_prompt)
from .data import demonstration_from_record as graph_demonstration, first_turn_from_record, load_graph
from .sft import Demonstration, decision_text
from .short_readout import correct_ids, diagnostic_prompt
from .transcript import encode_trajectory


def paired_turns(turn, style="full_ranking", pair=None):
    """Same user text and IDs, with original and swapped Q supervision."""
    swapped = swap_best_worst_q(turn.step) if pair is None else swap_candidate_q(turn.step, pair)
    if swapped is None:
        return ()
    if style == "full_ranking":
        return tuple(replace(turn, step=step,
                             answer_text=decision_text(step, step.map_minimal_candidates[0]),
                             chosen_id=step.map_minimal_candidates[0])
                     for step in (turn.step, swapped))
    if style not in ("pairwise", "nearest"):
        raise ValueError(f"Unknown readout style: {style}")
    if pair is None:
        pair = (turn.step.map_minimal_candidates[0],
                int(np.argmax(turn.step.candidate_map_distances)) + 1)
    user_text, tag = diagnostic_prompt(turn.user_text, style, pair)
    return tuple(replace(turn, step=step, user_text=user_text,
                         answer_text=f"<{tag}>{min(correct_ids(step, style, pair))}</{tag}>",
                         chosen_id=min(correct_ids(step, style, pair)))
                 for step in (turn.step, swapped))


def renumber_turn(turn, task, order):
    """Recompute the complete gold answer after moving action/Q pairs together."""
    step = reorder_candidates(turn.step, order)
    chosen = step.map_minimal_candidates[0]
    return replace(turn, step=step, chosen_id=chosen,
                   user_text=renumbered_prompt(task, step, turn.user_text),
                   answer_text=decision_text(step, chosen))


class CounterfactualFirstTurnDataset(Dataset):
    """Independent map readout condition; examples are not environment trajectories."""

    def __init__(self, records, qmap, tokenizer, config, source_root=None):
        self.tokenizer, self.config = tokenizer, config
        self.turns = []
        graphs = {}
        for record in records:
            if config["task"] == "blocks":
                demo = blocks_demonstration(qmap, record,
                    config["maximum_demonstration_actions"])
            else:
                graph_id = record["graph_id"]
                if graph_id not in graphs:
                    graphs[graph_id] = load_graph(source_root, graph_id)[:2]
                if "candidate_pair" in record:
                    turn = first_turn_from_record(*graphs[graph_id], record)
                    self.turns.extend(paired_turns(turn,
                        config["training"].get("readout_style", "full_ranking"),
                        pair=record["candidate_pair"]))
                    continue
                demo = graph_demonstration(*graphs[graph_id], record)
            if not demo.turns or demo.turns[0].step.done:
                continue
            self.turns.extend(paired_turns(demo.turns[0],
                config["training"].get("readout_style", "full_ranking")))
        if not self.turns or len(self.turns) % 2:
            raise ValueError("Counterfactual readout needs matched non-tie pairs")
        self.base_turns = tuple(self.turns)
        self.order_augmentation = config["training"].get("candidate_order_augmentation", "none")
        if self.order_augmentation not in ("none", "per_epoch"):
            raise ValueError("Unknown candidate order augmentation")
        if (self.order_augmentation != "none" and
                config["training"].get("readout_style", "full_ranking") != "full_ranking"):
            raise ValueError("Candidate augmentation currently requires full-ranking completion")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Both members share a deterministic permutation; resume reproduces it."""
        if self.order_augmentation == "none":
            return
        self.turns = []
        seed = self.config["training"].get("augmentation_seed", self.config["seed"])
        for index in range(0, len(self.base_turns), 2):
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(epoch), index // 2]))
            count = len(self.base_turns[index].step.candidate_actions)
            order = rng.permutation(count).tolist()
            self.turns.extend(renumber_turn(turn, self.config["task"], order)
                              for turn in self.base_turns[index:index + 2])

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, index):
        turn = self.turns[index]
        diagnostic = Demonstration((turn,), turn.executed_path, True)
        encoded = encode_trajectory(diagnostic, self.tokenizer,
            self.config["maximum_sequence_tokens"],
            chat_template_kwargs=self.config.get("chat_template_kwargs", {}))
        if self.config["training"].get("decision_focus_weight", 1) > 1:
            from .convergence import with_decision_mask
            return with_decision_mask((encoded, [turn.step]), self.tokenizer)
        return encoded, [turn.step]
