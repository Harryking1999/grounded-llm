"""Reuse task demonstrations and the common prepared-route loader."""
from functools import lru_cache
import numpy as np
from grounded_llm.core.sft.dataset import PreparedRoutes
from grounded_llm.core.sft.transcript import encode_trajectory
from grounded_llm.tasks.blocks.sft import renumber_trajectory
from grounded_llm.tasks.restoration_sft import environment, demonstration


class Routes(PreparedRoutes):
    def __init__(self, path, real_map, *, config, baseline, tokenizer):
        self.config, self.tokenizer = config, tokenizer
        renumber = config['baselines'][baseline].get('renumber_each_epoch', False)
        super().__init__(path, real_map, transform=self.augment if renumber else None)

    @lru_cache(None)
    def env(self, task):
        return environment(task, self.path / task / 'oracle.npy')

    def augment(self, row, item, epoch, index):
        if row.get('graph_sft'):
            from grounded_llm.tasks.state_graph_sft import demonstration as graph_demonstration
            rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'], epoch, index]))
            demo = graph_demonstration(row,item['steps'],rng)
            encoded = encode_trajectory(demo,self.tokenizer,self.config['maximum_sequence_tokens'])
            return encoded,[turn.step for turn in demo.turns]
        if row['task'] == 'blocks':
            demo = renumber_trajectory(row, item['steps'], self.config['seed'], epoch, index,
                                      'ranking_first')
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'], epoch, index]))
            demo = demonstration(row['task'], self.env(row['task']), row, item['steps'], rng)
        encoded = encode_trajectory(demo, self.tokenizer, self.config['maximum_sequence_tokens'])
        return encoded, [turn.step for turn in demo.turns]
