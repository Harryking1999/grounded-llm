"""One shared no-wall6x6 Qmap across vehicle layouts."""
from grounded_llm.core.qmap.transitions import TransitionQMap


class MapReader(TransitionQMap):
    def __init__(self, env, checkpoint, device='cpu'):
        if env.walls:
            raise ValueError('Adopted scope is no-wall6x6')
        super().__init__(env, checkpoint, 'rush_hour', device)
        encoding=self.config.get('input_encoding','identity')
        self._features=lambda states:env.features(states,encoding=encoding)
