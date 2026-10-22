"""One shared no-wall6x6 Qmap across vehicle layouts."""
from grounded_llm.core.qmap.transitions import TransitionQMap


class MapReader(TransitionQMap):
    def __init__(self, env, checkpoint, device='cpu'):
        if env.walls:
            raise ValueError('Adopted scope is no-wall6x6')
        super().__init__(env, checkpoint, 'rush_hour', device)
