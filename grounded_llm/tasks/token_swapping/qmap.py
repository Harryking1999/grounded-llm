"""Fixed-tree Qmap. Complete permutations, arbitrary query goals."""
from grounded_llm.core.qmap.transitions import TransitionQMap


class MapReader(TransitionQMap):
    def __init__(self, env, checkpoint, device='cpu'):
        super().__init__(env, checkpoint, 'token_swapping', device)
        if env.n != self.scope['vertex_count'] or list(map(list, env.edges)) != self.scope['edges']:
            raise ValueError('Checkpoint is bound to a different labelled tree')
