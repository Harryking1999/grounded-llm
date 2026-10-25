"""Weighted complete-goal map reader for one personnel count."""
from grounded_llm.core.qmap.transitions import TransitionQMap
from .environment import Bridge


class MapReader(TransitionQMap):
    def __init__(self,env,checkpoint,device='cpu'):
        super().__init__(env,checkpoint,'bridge',device,lambda a:env.actions[a])
        if self.scope['people']!=env.n:raise ValueError('Personnel count differs from trained map')
        self._features=Bridge(env.times,feature_encoding=self.config.get('input_encoding','identity')).features
