"""Check split disjointness and deterministic trajectory reconstruction."""

import unittest

from experiments.flamingo_map_reader.src.data import (
    demonstration_from_record, select_graph_records, shortest_move_counts,
)
from test_graph import line_graph


class DataSelectionTest(unittest.TestCase):
    def test_reserved_pair_and_reverse_never_enter_train_or_validation(self):
        environment, qmap = line_graph()
        self.assertEqual(shortest_move_counts(environment.adjacency)[0, 2], 2)
        config = {
            "ordinary_shortest_moves": [1],
            "train_pairs_per_length_per_graph": 1,
            "validation_pairs_per_length_per_graph": 1,
            "train_start_equals_goal_per_graph": 1,
            "validation_start_equals_goal_per_graph": 1,
            "maximum_demonstration_actions": 3,
        }
        suite = {"cases": [{"start": 0, "goal": 1}]}
        records, excluded = select_graph_records(
            environment, qmap, suite, "tiny", config, seed=17,
        )
        self.assertEqual(len(records), 4)
        self.assertEqual(excluded, {"dead_end": 0, "over_action_limit": 0})
        pairs = [(record["start"], record["goal"]) for record in records]
        self.assertEqual(len(set(pairs)), 4)
        self.assertNotIn((0, 1), pairs)
        self.assertNotIn((1, 0), pairs)
        repeat, _ = select_graph_records(environment, qmap, suite, "tiny",
                                         config, seed=17)
        self.assertEqual(records, repeat)
        for record in records:
            first = demonstration_from_record(environment, qmap, record)
            second = demonstration_from_record(environment, qmap, record)
            self.assertTrue(first.success)
            self.assertEqual(first.executed_path, second.executed_path)
            self.assertEqual([turn.answer_text for turn in first.turns],
                             [turn.answer_text for turn in second.turns])


if __name__ == "__main__":
    unittest.main()
