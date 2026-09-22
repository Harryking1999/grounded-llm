"""Persistent frozen LM replicas share one adapter graph and one optimizer."""
import copy
import torch
from torch import nn
from torch.nn.parallel import parallel_apply


def model_readout(model, batch, supervised):
    output = model.model(inputs_embeds=batch['inputs_embeds'],
                         attention_mask=batch['attention_mask'],
                         position_ids=batch['position_ids'], use_cache=False)
    if supervised:
        from .training import answer_losses
        return answer_losses(output.last_hidden_state, batch['labels'], model.lm_head)
    return model.lm_head(output.last_hidden_state[:, -1]).float()


class FrozenReplica(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, batch, supervised):
        device = self.model.get_input_embeddings().weight.device
        batch = {key: value.to(device) for key, value in batch.items()}
        return model_readout(self.model, batch, supervised)


class ParallelReadout:
    """Split examples, not model layers; peer copies keep gradients to the adapter."""
    def __init__(self, models):
        self.replicas = [FrozenReplica(model) for model in models]
        self.devices = [model.get_input_embeddings().weight.device for model in models]

    def __call__(self, batch, supervised):
        count = len(batch['inputs_embeds'])
        boundaries = torch.tensor_split(torch.arange(count), min(count, len(self.replicas)))
        shards = [{key: value[int(indices[0]):int(indices[-1]) + 1] for key, value in batch.items()}
                  for indices in boundaries]
        inputs = [(shard, supervised) for shard in shards]
        if all(device.type == 'cuda' for device in self.devices):
            outputs = parallel_apply(self.replicas[:len(shards)], inputs, devices=self.devices[:len(shards)])
        else:
            outputs = [replica(*args) for replica, args in zip(self.replicas, inputs)]
        return torch.cat([output.to(self.devices[0]) for output in outputs])


def configure_parallel_readout(interface, devices):
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError('Parallel readout requires distinct devices')
    resolved = [torch.device(device) for device in devices]
    if resolved[0] != interface.device or any(device.type != 'cuda' for device in resolved):
        raise ValueError('First CUDA readout device must own the adapter and primary model')
    models = [interface.model] + [copy.deepcopy(interface.model).to(device) for device in resolved[1:]]
    interface.readout_executor = ParallelReadout(models)
    interface.frozen_models = models


def readout(interface, batch, supervised):
    executor = getattr(interface, 'readout_executor', None)
    return executor(batch, supervised) if executor else model_readout(interface.model, batch, supervised)
