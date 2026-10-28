"""Reduce already-weighted replica gradients and perform one shared Adam step."""
import torch


def replica_step(models, optimizer, worker, pool, clip_norm):
    for model in models:
        model.train()
        model.zero_grad(set_to_none=True)
    values = list(pool.map(worker, range(len(models))))
    primary = models[0]
    for replica in models[1:]:
        for source, target in zip(replica.parameters(), primary.parameters()):
            target.grad.add_(source.grad.to(target.device))
    torch.nn.utils.clip_grad_norm_(primary.parameters(), clip_norm)
    optimizer.step()
    for replica in models[1:]:
        replica.load_state_dict(primary.state_dict())
    return values
