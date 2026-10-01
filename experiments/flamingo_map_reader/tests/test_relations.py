import unittest
from types import SimpleNamespace

import torch

from experiments.flamingo_map_reader.src.evaluate_graph_readout import generate_answer
from experiments.flamingo_map_reader.src.relations import parse_ranking, score_relationships
from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.sft import decision_text
from experiments.flamingo_map_reader.src.summarize_graph_eval import score_rollout_relationships
from test_graph import line_graph
from test_graph_eval import ByteTokenizer


class RelationshipTests(unittest.TestCase):
    def test_current_and_candidates_are_scored_together(self):
        step = SimpleNamespace(current_map_distance=1.5,
            candidate_map_distances=(2.0, 1.0), candidate_actions=(4, 7),
            map_minimal_candidates=(2,), done=False)
        score = score_relationships(step,
            "Map-distance ranking to the goal, closest to farthest: 2 < current < 1.\n"
            "<action>2</action>")
        self.assertEqual(score["pairwise_correct"], 3)
        self.assertTrue(score["exact_ranking"])
        self.assertTrue(score["closest_candidate_set_exact"])
        self.assertTrue(score["action_follows_ranking"])

    def test_missing_slot_fails_instead_of_dropping_a_pair(self):
        step = SimpleNamespace(current_map_distance=1.5,
            candidate_map_distances=(2.0, 1.0), candidate_actions=(4, 7),
            map_minimal_candidates=(2,), done=False)
        score = score_relationships(step,
            "Map-distance ranking to the goal, closest to farthest: 2 < 1.\n"
            "<action>2</action>")
        self.assertFalse(score["valid_ranking"])
        self.assertEqual(score["pairwise_correct"], 0)
        self.assertEqual(score["pairwise_total"], 3)

    def test_true_tie_is_counted(self):
        step = SimpleNamespace(current_map_distance=1.0,
            candidate_map_distances=(1.0, 2.0), candidate_actions=(4, 7),
            map_minimal_candidates=(1,), done=False)
        score = score_relationships(step,
            "Map-distance ranking to the goal, closest to farthest: 1 = current < 2.\n"
            "<action>1</action>")
        self.assertTrue(score["exact_ranking"])
        self.assertEqual(parse_ranking("Map-distance ranking to the goal, closest to farthest: 1 < 1 < current.", 2), None)

    def test_wrong_relation_and_action_are_separate(self):
        step = SimpleNamespace(current_map_distance=1.5,
            candidate_map_distances=(2.0, 1.0), candidate_actions=(4, 7),
            map_minimal_candidates=(2,), done=False)
        score = score_relationships(step,
            "Map-distance ranking to the goal, closest to farthest: 1 < current < 2 .\n"
            "<action>2</action>")
        self.assertTrue(score["valid_ranking"])
        self.assertEqual(score["pairwise_correct"], 0)
        self.assertFalse(score["closest_candidate_set_exact"])
        self.assertTrue(score["action_map_minimum"])
        self.assertFalse(score["action_follows_ranking"])

    def test_replay_matches_actual_candidate_order_and_path(self):
        environment, qmap = line_graph()
        case = {"id": "case_0", "start": 0, "goal": 1}
        step = graph_step(environment, qmap, 0, 1, executed_path=[0])
        answer = decision_text(step, 1)
        row = {"graph_id": "graph_test", "case_id": "case_0", "path": [0, 1],
               "trace": [{"current": 0, "candidate_destinations": list(step.candidate_destinations),
                          "answer": answer, "chosen_id": 1, "after": 1}]}
        scored = score_rollout_relationships(row, environment, qmap, case, seed=1)
        self.assertEqual(len(scored), 1)
        self.assertTrue(scored[0]["exact_ranking"])
        row["trace"][0]["candidate_destinations"] = [99]
        with self.assertRaises(ValueError):
            score_rollout_relationships(row, environment, qmap, case, seed=1)

    def test_fixed_turn_generation_obeys_readout_budget(self):
        class Reader:
            def generate(self, timeline, input_ids, **kwargs):
                self.maximum_new_tokens = kwargs["max_new_tokens"]
                answer = list(b"<action>1</action>")
                return torch.cat((input_ids,
                                  torch.tensor([answer], device=input_ids.device)), dim=1)

        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 0, 1, executed_path=[0])
        reader = Reader()
        answer = generate_answer(reader, ByteTokenizer(), step, "hello",
            {"maximum_sequence_tokens": 2000,
             "evaluation": {"action_max_new_tokens": 2048}}, torch.device("cpu"), 512)
        self.assertEqual(answer, "<action>1</action>")
        self.assertEqual(reader.maximum_new_tokens, 512)


if __name__ == "__main__":
    unittest.main()
