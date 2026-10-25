"""Hard-layout Qmaps, loaded independently of oracle data."""
from grounded_llm.core.qmap.transitions import TransitionQMap


class MapReader(TransitionQMap):
    def __init__(self, env, checkpoint, device='cpu'):
        if (env.height, env.width, len(env.goals)) != (10, 10, 4):
            raise ValueError('Hard Qmap requires10x10 and four boxes/targets')
        super().__init__(env, checkpoint, 'sokoban', device, lambda a: env.actions[a])
        if self.scope.get('shared_across_layouts') is False and self.scope.get('floor')!=sorted(env.floor):
            raise ValueError('This Qmap was trained for a different fixed wall layout/orientation')
        if 'goal_markers' in self.scope and self.scope['goal_markers']!=sorted(env.goals):
            raise ValueError('Keep the fixed board markers; express each complete target as the query goal')
