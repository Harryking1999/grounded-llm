"""Checks for goal-dependent labels and grouped supervision."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.blocks_distance_map.src.multiboard_data import candidate_pairs, choose_landmarks, contrast_cases
from experiments.blocks_distance_map.src.multiboard_eval import listwise_groups
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class MultiboardTests(unittest.TestCase):
    def test_xor_query_retains_physical_successors_and_original_q_width(self):
        from grounded_llm.tasks.blocks.model import BoardEncoder, goal_relative_boards
        from grounded_llm.tasks.blocks.qmap import FrozenBoardMap, blocks_step
        from grounded_llm.tasks.blocks.oracle import successors
        from grounded_llm.tasks.blocks.rules import PLACEMENTS
        torch.manual_seed(8)
        model = BoardEncoder(128, 16, 13, 'directed_sum')
        raw = FrozenBoardMap(model)
        relative = FrozenBoardMap(model, coordinate_frame='goal_xor')
        for source in range(64):
            for goal in range(64):
                encoded = goal_relative_boards([source], goal)[0]
                self.assertEqual(encoded ^ goal, source)
                if source & goal == goal:
                    self.assertEqual(encoded, source & ~goal)
                for action, child in successors(source):
                    self.assertEqual(child ^ goal, encoded ^ PLACEMENTS[action][0])
        # A legal removal can erase goal cells; it still has a finite, full Q slot.
        source, goal = 7, 4
        step = blocks_step(relative, source, goal)
        self.assertEqual(list(zip(step.candidate_actions, step.candidate_destinations)), successors(source))
        self.assertEqual(step.map_batch.vectors.shape[-1], 128)
        step.map_batch.validate(1000)
        torch.testing.assert_close(step.map_batch.vectors[0], raw.encode([source ^ goal, 0, *[s ^ goal for _, s in successors(source)]]))
        self.assertTrue(np.isfinite(step.candidate_map_distances).all())
        for i, pair in enumerate(successors(source), 1):
            self.assertEqual(step.execute(i), pair)
        for source in (0, 7, 31, 63):
            a, b = blocks_step(raw, source, 0), blocks_step(relative, source, 0)
            torch.testing.assert_close(a.map_batch.vectors, b.map_batch.vectors, rtol=0, atol=0)
            self.assertEqual(a.candidate_map_distances, b.candidate_map_distances)
        self.assertTrue(blocks_step(relative, goal, goal).done)
        self.assertFalse(blocks_step(relative, 0, goal).done)
        from grounded_llm.core.qmap.reader import CachedQMap
        for reader in (raw, relative):
            cached = CachedQMap(reader)
            for target in (0, goal, 3, goal):
                actual, expected = blocks_step(cached, 7, target), blocks_step(reader, 7, target)
                torch.testing.assert_close(actual.map_batch.vectors, expected.map_batch.vectors)
                self.assertEqual(actual.candidate_map_distances, expected.candidate_map_distances)
        from grounded_llm.tasks.state_graph_sft import StateTask
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'q.pt'
            torch.save(dict(config={'model': dict(state_dim=128, encoder_hidden_dim=16)},
                            cap=13, metric=model.metric, model=model.state_dict()), path)
            task = StateTask(dict(task='blocks', map_checkpoint='q.pt',
                                  context={'coordinate_frame': 'goal_xor'}), temp)
            flan_step = task.make_step(7, goal)
            torch.testing.assert_close(flan_step.map_batch.vectors, step.map_batch.vectors)
            self.assertEqual(flan_step.candidate_destinations, step.candidate_destinations)

    def test_residual_query_preserves_clearing_and_goal_preserving_transitions(self):
        from experiments.blocks_distance_map.src.tree_supervision_eval import QScorer
        from grounded_llm.tasks.blocks.model import BoardEncoder
        from grounded_llm.tasks.blocks.oracle import successors
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'q.pt';model=BoardEncoder(8,16,13,'directed_sum')
            torch.save(dict(config={'model':dict(state_dim=8,encoder_hidden_dim=16)},
                cap=13,metric=model.metric,model=model.state_dict(),source_commit='fixture',step=0),path)
            raw=QScorer(path,'cpu');relative=QScorer(path,'cpu',coordinate_frame='goal_residual')
            masks=[63,31,47,0];goals=[3,3,15,1]
            batch=relative.scores(masks,goals)
            for i,(source,goal) in enumerate(zip(masks,goals)):
                np.testing.assert_array_equal(batch[i:i+1],relative.scores([source],goal))
            for source in range(64):
                children=[s for _,s in successors(source)]
                np.testing.assert_array_equal(relative.scores(children,0),raw.scores(children,0))
                for goal in range(64):
                    if source&goal!=goal:continue
                    preserved=[s for s in children if s&goal==goal]
                    self.assertEqual({s^goal for s in preserved},{s for _,s in successors(source^goal)})
                    scores=relative.scores(children,goal)
                    valid=np.array([s&goal==goal for s in children],dtype=bool)
                    self.assertTrue(np.isinf(scores[~valid]).all())
                    np.testing.assert_array_equal(scores[valid],raw.scores([s^goal for s in preserved],0))
                    if len(children) and len(preserved):self.assertIn(children[int(scores.argmin())],preserved)

    def test_dead_state_can_reach_a_nonempty_dead_goal(self):
        oracle = DistanceOracle()
        source, goal = 0b100011, 0b100000
        self.assertEqual(oracle.distance(source, 0), -1)
        self.assertEqual(oracle.distance(goal, 0), -1)
        self.assertEqual(oracle.distance(source, goal), 1)
        pairs = candidate_pairs([[source, goal]], [goal], 2, np.random.default_rng(1))
        self.assertIn((source, goal), pairs)

    def test_isolated_landmark_is_selected_explicitly(self):
        isolated = 0b100000
        self.assertEqual(choose_landmarks([[0b100011, isolated]], 1,
                                          np.random.default_rng(1)), [isolated])

    def test_heldout_landmarks_include_isolated_and_supported(self):
        isolated, supported = 0b100000, 0b111000
        landmarks = choose_landmarks([[0b111111, supported, 0], [0b100011, isolated]],
                                     2, np.random.default_rng(1), heldout_count=2)
        self.assertEqual(set(landmarks), {isolated, supported})

    def test_contrast_reverses_with_goal(self):
        oracle = DistanceOracle()
        cases = contrast_cases([[0b11111111, 0b111111, 0]], oracle, 1, np.random.default_rng(2))
        self.assertTrue(cases)
        _, a, b, ga, gb = cases[0]
        self.assertEqual(oracle.distance(a, ga), 1)
        self.assertEqual(oracle.distance(b, gb), 1)
        self.assertEqual(oracle.distance(b, ga), -1)
        self.assertEqual(oracle.distance(a, gb), -1)

    def test_listwise_groups_share_an_anchor_and_different_labels(self):
        pairs = np.asarray([[1, 3, 1, 0], [2, 3, 2, 0], [4, 3, -1, 2],
                            [1, 5, 1, 0], [1, 6, -1, 1]], dtype=np.int32)
        groups = listwise_groups(pairs, 3, per_anchor=2)
        self.assertGreater(len(groups), 0)
        for group in groups:
            selected = pairs[group]
            self.assertTrue(len(set(selected[:, 0])) == 1 or len(set(selected[:, 1])) == 1)
            self.assertGreater(len(set(selected[:, 2])), 1)


if __name__ == "__main__":
    unittest.main()
