"""Plain full-parameter language-model loss, without any map components."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset


class RouteDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path/'dataset.json').read_text())
        if not (self.path/'ready.json').exists():
            raise ValueError('Dataset has not finished preparation')
        self.rows = self.metadata['rows']
        self.ids = np.memmap(self.path/'input_ids.bin', dtype='<i4', mode='r')
        self.labels = np.memmap(self.path/'labels.bin', dtype='<i4', mode='r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        section = slice(row['offset'], row['offset'] + row['length'])
        quota = self.metadata['tasks'][row['task_index']]['quota']
        weight = len(self.rows) / (len(self.metadata['tasks']) * quota)
        return dict(input_ids=torch.tensor(self.ids[section].copy(), dtype=torch.long),
                    labels=torch.tensor(self.labels[section].copy(), dtype=torch.long),
                    sample_weight=weight)


class Collator:
    def __init__(self, pad_token):
        self.pad_token = pad_token

    def __call__(self, rows):
        maximum = max(len(row['input_ids']) for row in rows)
        ids = torch.full((len(rows), maximum), self.pad_token, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            size = len(row['input_ids'])
            ids[index, :size], labels[index, :size] = row['input_ids'], row['labels']
            attention[index, :size] = 1
        return dict(input_ids=ids, labels=labels, attention_mask=attention,
                    sample_weight=torch.tensor([r['sample_weight'] for r in rows]))


class SupervisedLM(nn.Module):
    def __init__(self, base, chunk_tokens=128):
        super().__init__()
        self.base, self.chunk_tokens = base, chunk_tokens

    def forward(self, input_ids, attention_mask, labels, sample_weight):
        hidden = self.base.model(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=False).last_hidden_state[:, :-1]
        targets = labels[:, 1:]
        selected = targets.ne(-100)
        count = selected.sum(dim=1)
        if torch.any(count == 0):
            raise ValueError('A trajectory has no predicted supervised tokens')
        token_weights = (sample_weight / count / input_ids.shape[0])[:, None].expand_as(targets)[selected]
        features, targets = hidden[selected], targets[selected]

        def token_loss(features, targets, weights):
            logits = self.base.lm_head(features).float()
            return (nn.functional.cross_entropy(logits, targets, reduction='none') * weights).sum()

        loss = hidden.new_zeros((), dtype=torch.float32)
        for offset in range(0, len(targets), self.chunk_tokens):
            section = slice(offset, offset + self.chunk_tokens)
            if self.training and torch.is_grad_enabled():
                loss = loss + checkpoint(token_loss, features[section], targets[section],
                                         token_weights[section], use_reentrant=False)
            else:
                loss = loss + token_loss(features[section], targets[section], token_weights[section])
        return loss
