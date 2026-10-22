"""One shared hard-layout Qmap, loaded independently of oracle data."""
from grounded_llm.core.qmap.transitions import TransitionQMap


class MapReader(TransitionQMap):
    def __init__(self, env, checkpoint, device='cpu'):
        if (env.height, env.width, len(env.goals)) != (10, 10, 4):
            raise ValueError('Hard Qmap requires10x10 and four boxes/targets')
        super().__init__(env, checkpoint, 'sokoban', device, lambda a: env.actions[a])
