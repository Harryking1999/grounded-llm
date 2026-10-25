"""Weighted complete-goal map reader for one personnel count."""
from grounded_llm.core.qmap.transitions import TransitionQMap


class MapReader(TransitionQMap):
    def __init__(self,env,checkpoint,device='cpu'):
        super().__init__(env,checkpoint,'bridge',device)
        if self.scope['people']!=env.n:raise ValueError('Personnel count differs from trained map')
