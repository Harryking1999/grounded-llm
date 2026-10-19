"""Frozen Q inference; callers supply task features and checkpoint construction."""
import numpy as np
import torch


class FrozenQMap:
    def __init__(self, encoder, features, device="cpu", *, cpu_output=False):
        self.encoder = encoder.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.metric = encoder.metric
        self._features = features
        self.cpu_output = cpu_output

    @property
    def model(self):
        return self.encoder

    @torch.no_grad()
    def encode(self, states):
        features = torch.as_tensor(self._features(states), device=self.device)
        vectors = self.encoder.encode(features)
        return vectors.cpu() if self.cpu_output else vectors

    def distance(self, source, goal):
        return self.encoder.distance(source, goal)

    @torch.no_grad()
    def scores(self, source, goal):
        return self.distance(self.encode(source), self.encode(goal)).cpu().numpy()


class EnvironmentQMap(FrozenQMap):
    """Inference for tasks exposing features, successors and an action catalogue."""
    def __init__(self, env, encoder, device="cpu"):
        self.env = env
        super().__init__(encoder, env.features, device)

    def choose(self, state, goal):
        children = self.env.successors([state])[0]
        legal = np.flatnonzero(children >= 0)
        if not len(legal):
            raise ValueError("Cannot choose an action without legal successors")
        action = int(legal[self.scores(children[legal], [goal]).argmin()])
        return self.env.actions[action], int(children[action])


class CachedQMap:
    def __init__(self,qmap):
        self.qmap,self.cache,self.metric=qmap,{},qmap.metric
    def encode(self,states):
        missing=list(dict.fromkeys(s for s in states if s not in self.cache))
        if missing:
            vectors=self.qmap.encode(missing)
            self.cache.update(zip(missing,vectors))
        return torch.stack([self.cache[s] for s in states])
