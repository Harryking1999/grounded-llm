import json
from pathlib import Path
import unittest

import numpy as np

from experiments.cml_map_scaling.src.core import shortest_distances
from experiments.cml_map_scaling.src.step2_data import make_split
from experiments.cml_map_scaling.src.step2_action_pilot import make_training_data


class ActionPilotDataTest(unittest.TestCase):
    def test_training_labels_stay_in_train_and_replay_covers_all_roles(self):
        root = Path('experiments/cml_map_scaling/configs')
        config = json.loads((root / 'step2.json').read_text())
        spec = json.loads((root / 'step2_action_pilot.json').read_text())
        split = make_split(config)
        graph = json.loads(Path(config['assets']['graph_reference']).read_text())
        adj = np.zeros((32, 32), dtype=bool)
        for u, neighbors in graph['neighbors'].items():
            adj[int(u), neighbors] = True
        distances = shortest_distances(adj)
        data = make_training_data(spec, split, adj, distances)
        self.assertEqual(len(data['actions']), 80)
        self.assertEqual(len(data['reports']), 80)
        train_pairs = set(map(tuple, split['train']))
        for row in data['actions'] + data['reports']:
            self.assertIn(tuple(sorted((row['u'], row['g']))), train_pairs)
        for row in data['actions']:
            self.assertTrue(adj[row['u'], row['target']])
            self.assertLess(distances[row['target'], row['g']], distances[row['u'], row['g']])
        for task, role in (('report_current', 'u'), ('report_goal', 'g')):
            self.assertEqual({r[role] for r in data['reports'] if r['task'] == task}, set(range(32)))
        scaled = json.loads((root / 'step2_action_scaling.json').read_text())
        same_size = make_training_data(scaled, split, adj, distances)
        self.assertEqual(len(same_size['actions']), 80)
        self.assertEqual(len(same_size['reports']), 80)


if __name__ == '__main__':
    unittest.main()
