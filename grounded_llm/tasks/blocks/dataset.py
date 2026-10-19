"""Board trajectory augmentation for the common prepared SFT dataset."""
from grounded_llm.core.sft.dataset import PreparedRoutes
from grounded_llm.core.sft.transcript import encode_trajectory
from .sft import renumber_trajectory


class Routes(PreparedRoutes):
    def __init__(self, path, real_map, *, config=None, baseline=None, tokenizer=None):
        self.config, self.baseline, self.tokenizer = config, baseline, tokenizer
        self.renumber = bool(config and config['baselines'][baseline].get('renumber_each_epoch', False))
        if self.renumber and (not real_map or tokenizer is None):
            raise ValueError('Epoch renumbering requires a map trajectory and tokenizer')
        super().__init__(path, real_map, transform=self.renumber_item if self.renumber else None)

    def renumber_item(self, row, item, epoch, index):
        demo = renumber_trajectory(row, item['steps'], self.config['seed'], epoch, index,
            self.config['baselines'][self.baseline].get('response_order', 'reason_first'))
        encoded = encode_trajectory(demo, self.tokenizer, self.config['maximum_sequence_tokens'])
        return encoded, [turn.step for turn in demo.turns]
