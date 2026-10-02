"""Paired first-turn Q interventions for separate interface-readout training."""

from dataclasses import replace

import numpy as np

from torch.utils.data import Dataset

from .blocks_data import demonstration_from_record as blocks_demonstration
from .counterfactual import swap_best_worst_q, swap_candidate_q
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

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, index):
        turn = self.turns[index]
        diagnostic = Demonstration((turn,), turn.executed_path, True)
        encoded = encode_trajectory(diagnostic, self.tokenizer,
            self.config["maximum_sequence_tokens"],
            chat_template_kwargs=self.config.get("chat_template_kwargs", {}))
        return encoded, [turn.step]
