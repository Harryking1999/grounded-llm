"""Complete exact-rational goals, using the trained operand capacity."""
from grounded_llm.core.qmap.transitions import TransitionQMap
from .environment import Game24


class MapReader(TransitionQMap):
    def __init__(self,env,checkpoint,device='cpu'):
        super().__init__(env,checkpoint,'game24',device,lambda a:env.actions[a])
        if self.scope['max_numbers']!=env.max_numbers:raise ValueError('Operand capacity differs from trained map')
        self._features=Game24(env.max_numbers,feature_encoding=self.config.get('input_encoding','identity')).features
