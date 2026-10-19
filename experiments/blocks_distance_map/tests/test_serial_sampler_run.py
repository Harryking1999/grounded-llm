import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.blocks_distance_map.src.full_graph_data import local_tiles
from experiments.blocks_distance_map.src.full_graph_run import run
from experiments.blocks_distance_map.src.model import BoardEncoder
from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.blocks_distance_map.src.symbolic_graph import build_symbolic, SymbolicComponent


class SerialSamplerRunTest(unittest.TestCase):
    def test_both_shards_and_adam_resume_identically_on_one_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = [0, 1, 10]
            build_symbolic(3, local_tiles(cells), root / 'data/graphs/0', 10000)
            graph = SymbolicComponent(root / 'data/graphs/0')
            record = {'initial': '1027', 'construction': [{}],
                      'components': [{'graph': 0, 'cells': cells}]}
            (root / 'data/roots.json').write_text(json.dumps([record, record]))
            (root / 'data/summary.json').write_text(json.dumps({'complete': True,
                'total_root_local_pairs': 2 * graph.size ** 2, 'total_root_local_states': 2 * graph.size}))
            masks = [sum(((int(m) >> i) & 1) << c for i, c in enumerate(cells))
                     for m in graph.states[np.arange(graph.size)]]
            oracle = DistanceOracle()
            rows = [[i, j, oracle.distance(s, g), int(oracle.distance(s, g) < 0)]
                    for i, s in enumerate(masks) for j, g in enumerate(masks)]
            np.savez(root / 'legacy.npz', states=np.asarray(masks, dtype=str), train=np.asarray(rows))
            settings = {'state_dim': 8, 'encoder_hidden_dim': 16}
            model = BoardEncoder(8, 16, 13, 'directed_sum')
            optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
            options = {'seed': 18, 'boards_per_batch': 2, 'legacy_fraction': .25,
                       'broad_fraction': .25, 'clear_fraction': .5, 'expanded_stream': True,
                       'learning_rate': .0003, 'temperature': 1., 'listwise_weight': 1.,
                       'original_cap': 13, 'unreachable_label': 19, 'cpu_threads': 1,
                       'steps': 4, 'batch_size_per_rank': 32, 'rank_contexts_per_rank': 16,
                       'log_every': 100, 'checkpoint_every': 2, 'sampler_shards': 2}
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': {'model': settings, 'training': options}, 'step': 0,
                        'cap': 13, 'metric': 'directed_sum'}, root / 'parent.pt')
            config = {'parent_checkpoint': str(root / 'parent.pt'), 'graph_data': str(root / 'data'),
                      'legacy_data': str(root / 'legacy.npz'), 'training': options}
            with patch('torch.cuda.is_available', return_value=False):
                run(config, root / 'resumed', 'test', smoke_steps=2)
                run(config, root / 'resumed', 'test', resume=root / 'resumed/checkpoints/step_000002.pt')
                run(config, root / 'continuous', 'test')
            a = torch.load(root / 'resumed/checkpoints/step_000004.pt', weights_only=False)
            b = torch.load(root / 'continuous/checkpoints/step_000004.pt', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            for shard in (0, 1):
                state = torch.load(root / f'resumed/sampler_000004_rank{shard}.pt', weights_only=False)
                self.assertGreater(state['pair_counts']['clear'], 0)
                self.assertEqual(int(next(iter(a['optimizer']['state'].values()))['step']), 4)
            exposure = json.loads((root / 'resumed/exposure_000004.json').read_text())
            self.assertEqual(exposure['physical_world_size'], 1)
            self.assertEqual(len(exposure['serialized_sampler_shard_reports']), 2)
            self.assertGreater(exposure['global_stream_phase']['estimated_distinct_relations'], 0)


if __name__ == '__main__':
    unittest.main()
