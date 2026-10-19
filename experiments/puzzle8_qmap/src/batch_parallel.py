"""Split one globally sampled batch across replicas of the same Puzzle8 map."""
from concurrent.futures import ThreadPoolExecutor
import copy

import numpy as np
import torch
from torch.nn import functional as F

from grounded_llm.tasks.puzzle8.model import soft_rank_loss


class BatchParallel:
    def __init__(self, model, features, devices):
        if not devices or str(features.device) != str(torch.device(devices[0])):
            raise ValueError("The first replica must use the optimizer's existing device")
        self.models = [model] + [copy.deepcopy(model).to(device) for device in devices[1:]]
        self.features = [features] + [features.to(device) for device in devices[1:]]
        self.pool = ThreadPoolExecutor(max_workers=len(devices))

    def update(self, rows, tasks, children, labels, optimizer, rank_weight, temperature):
        if min(len(rows), len(tasks)) < len(self.models):
            raise ValueError("Every replica needs both distance and ranking examples")
        partitions = [np.array_split(values, len(self.models)) for values in (rows, tasks, children, labels)]
        for model in self.models:
            model.train()
            model.zero_grad(set_to_none=True)

        def worker(index):
            model, features = self.models[index], self.features[index]
            device = features.device
            pair, task, child, truth = [parts[index] for parts in partitions]
            encoded = model.encode(features)
            def q(ids):
                return encoded[torch.as_tensor(ids, device=device, dtype=torch.long)]
            pair_loss = F.smooth_l1_loss(model.distance(q(pair[:, 0]), q(pair[:, 1])),
                                        torch.as_tensor(pair[:, 2], device=device, dtype=torch.float32))
            scores = model.distance(q(np.maximum(child, 0)), q(task[:, 1])[:, None])
            rank_loss = soft_rank_loss(scores, torch.as_tensor(truth, device=device),
                                       torch.as_tensor(child >= 0, device=device), temperature)
            # Each original mean uses its own full global denominator. Unequal
            # shard sizes therefore preserve both original loss weights.
            pair_part = pair_loss * (len(pair) / len(rows))
            rank_part = rank_loss * (len(task) / len(tasks))
            (pair_part + rank_weight * rank_part).backward()
            return float(pair_part.detach()), float(rank_part.detach())

        values = list(self.pool.map(worker, range(len(self.models))))
        primary = self.models[0]
        for replica in self.models[1:]:
            for source, target in zip(replica.parameters(), primary.parameters()):
                target.grad.add_(source.grad.to(target.device))
        # Clip and update once, after the complete original global gradient.
        torch.nn.utils.clip_grad_norm_(primary.parameters(), 5.)
        optimizer.step()
        for replica in self.models[1:]:
            replica.load_state_dict(primary.state_dict())
        return tuple(torch.tensor(sum(value[k] for value in values), device=self.features[0].device,
                                  dtype=self.features[0].dtype)
                     for k in range(2))

    def close(self):
        self.pool.shutdown()
