"""Prepared trajectories: maps stored once, fixed numbered variants reused each epoch."""

from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .blocks import BlocksStep
from .graph import GraphStep
from .memory import MapBatch
from .sft import Demonstration, SupervisedTurn
from .transcript import EncodedTrajectory, encode_trajectory
from .trajectory_protocol import numbering_plans, renumber_demonstration


def pack_demo(demo):
    rows = []
    for turn in demo.turns:
        step = {field.name: getattr(turn.step, field.name) for field in fields(turn.step)
                if field.name != "map_batch"}
        step["map_batch"] = {field.name: getattr(turn.step.map_batch, field.name).cpu()
                             for field in fields(turn.step.map_batch)}
        rows.append(dict(user_text=turn.user_text, answer_text=turn.answer_text,
                         executed_path=turn.executed_path, chosen_id=turn.chosen_id,
                         supervise=turn.supervise, step=step))
    return dict(turns=rows, executed_path=demo.executed_path, success=demo.success,
                no_solution=demo.no_solution)


def unpack_demo(value, task):
    cls = GraphStep if task == "graph" else BlocksStep
    turns = []
    for row in value["turns"]:
        step = dict(row["step"])
        step["map_batch"] = MapBatch(**step["map_batch"])
        turns.append(SupervisedTurn(**dict(row, step=cls(**step))))
    return Demonstration(tuple(turns), tuple(value["executed_path"]), value["success"],
                         value.get("no_solution", False))


def prepare_record(demo, record, config, tokenizer, output):
    plans = numbering_plans(demo, config["data"]["numbering_variants"], record["sample_seed"] + 719)
    encodings, lengths = [], []
    for plan in plans:
        numbered = renumber_demonstration(demo, plan, config["task"],
            config["data"].get("reported_candidates"))
        if demo.success or demo.no_solution:
            encoded = encode_trajectory(numbered, tokenizer,
                config["maximum_sequence_tokens"] if record["split"] == "train" else 10**8,
                chat_template_kwargs=config.get("chat_template_kwargs", {}))
            lengths.append(len(encoded.input_ids))
            # Evaluation variants need no persisted token copies.
            if record["split"] == "train":
                encodings.append(dict(input_ids=torch.tensor(encoded.input_ids, dtype=torch.int32),
                    labels=torch.tensor(encoded.labels, dtype=torch.int32),
                    token_map_ids=torch.tensor(encoded.token_map_ids, dtype=torch.int16),
                    answer_tokens=encoded.answer_tokens))
    torch.save(dict(demo=pack_demo(demo), plans=plans, encodings=encodings), output)
    return dict(record, prepared_file=output.name, variants=len(plans),
        max_tokens=max(lengths, default=0), action_count=len(demo.executed_path) - 1,
        greedy_success=demo.success, decision_turns=sum(not t.step.done for t in demo.turns))


def load_record(root, record, config, variant=0):
    value = torch.load(Path(root) / record["prepared_file"], weights_only=True, map_location="cpu")
    demo = unpack_demo(value["demo"], config["task"])
    return renumber_demonstration(demo, value["plans"][variant], config["task"],
                                 config["data"].get("reported_candidates"))


class PreparedTrajectoryDataset(Dataset):
    def __init__(self, manifest_path, records, config):
        self.root, self.config = Path(manifest_path).parent / "trajectories", config
        self.index = [(record, i) for record in records for i in range(record["variants"])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        record, variant = self.index[index]
        value = torch.load(self.root / record["prepared_file"], weights_only=True, map_location="cpu")
        raw = value["encodings"][variant]
        encoded = EncodedTrajectory(**{key: tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor
                                      for key, tensor in raw.items()})
        demo = renumber_demonstration(unpack_demo(value["demo"], self.config["task"]),
            value["plans"][variant], self.config["task"],
            self.config["data"].get("reported_candidates"))
        return encoded, [turn.step for turn in demo.turns]
