"""Frozen physical-state readers. No graph, label table or oracle at inference."""
import torch
from .reader import FrozenQMap
from .model import StateEncoder
from grounded_llm.core.reader.memory import q_map_batch


class TransitionQMap(FrozenQMap):
    def __init__(self, env, checkpoint, task, device='cpu', action_name=lambda action: action):
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get('model_type') != 'exact_state_pairs' or saved.get('task') != task:
            raise ValueError('Expected the matching exact-state-pair checkpoint')
        model = StateEncoder(input_dim=saved['input_dim'], **saved['config']['model']).to(device)
        model.load_state_dict(saved['model'])
        self.env, self.action_name, self.scope = env, action_name, saved['scope']
        self.config=saved['config']
        self.step = saved['step']
        super().__init__(model, env.features, device)

    @torch.no_grad()
    def map_batch(self, current, goal, rng=None):
        self.env.validate(current)
        self.env.validate(goal)
        candidates = list(self.env.transitions(current))
        if rng is not None:
            candidates = [candidates[i] for i in rng.permutation(len(candidates))]
        states = [current, goal, *[child for _, child, _ in candidates]]
        batch = q_map_batch(self.encode(states).cpu())
        return batch, tuple(self.action_name(a) for a, _, _ in candidates)

    @torch.no_grad()
    def choose(self, current, goal):
        candidates = list(self.env.transitions(current))
        if not candidates:
            raise ValueError('No legal successor')
        scores = self.scores([child for _, child, _ in candidates], [goal])
        choice = min(range(len(candidates)), key=lambda i: candidates[i][2]+float(scores[i]))
        action, child, _ = candidates[choice]
        return self.action_name(action), child
