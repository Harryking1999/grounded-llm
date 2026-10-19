"""Load prepared trajectories; task-specific transformations are injected."""
import json
from dataclasses import fields
from pathlib import Path
import torch
from torch.utils.data import Dataset


class PreparedRoutes(Dataset):
    def __init__(self, path, real_map, *, transform=None):
        self.path = Path(path)
        self.meta = json.loads((self.path / 'dataset.json').read_text())
        self.rows = self.meta['rows']
        self.key = 'map' if real_map else 'plain'
        self.epoch = 0
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = torch.load(self.path / 'trajectories' / row['prepared_file'],
                          map_location='cpu', weights_only=False)[self.key]
        if self.transform is not None:
            return self.transform(row, item, self.epoch, index)
        return item['encoded'], item['steps']


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
