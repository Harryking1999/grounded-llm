"""Checkpoint identity comes from published metadata, not spacing of saved steps."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.flamingo_map_reader.src.collect_evidence import (
    audit, checkpoint_epochs, complete_rollout_reachability,
)
from experiments.flamingo_map_reader.src.trajectory_eval import closed_loop
from test_graph import line_graph
from test_reachability import task_environment


class EvidenceTest(unittest.TestCase):
    def test_replay_includes_budget_choice_and_failed_answers_in_solvable_denominator(self):
        environment = task_environment(*line_graph())
        record = dict(trajectory_id="tiny", start=0, goal=2, sample_seed=7, shortest_moves=2)
        manifest = dict(config={}, records=[record])
        cases = []
        for answer in ("<action>1</action>", "<done/>", "<action>99</action>", "bad output"):
            session = Mock()
            session.ask.return_value = (answer, dict(context_exhausted=False))
            case = closed_loop(session, record, "", environment,
                               dict(maximum_demonstration_actions=0, data={}), 0)
            case.update(mode="rollout", trajectory_id="tiny", variant=0)
            self.assertEqual(case["moves"], 0)
            self.assertNotIn("action_keeps_goal_reachable", case["turns"][0])
            cases.append(case)
        # Neither terminal nor already-unsolvable states require reconstruction.
        cases.append(dict(mode="rollout", turns=[dict(done=True, remaining_shortest=0),
                                                dict(done=False, remaining_shortest=-1)]))
        with patch("experiments.flamingo_map_reader.src.trajectory_eval.TaskEnvironment",
                   return_value=environment):
            self.assertEqual(complete_rollout_reachability(cases, manifest), 4)
            self.assertEqual(complete_rollout_reachability(cases, manifest), 0)
        result = audit(cases)
        self.assertEqual(result["current_reachable_turns"], 4)
        self.assertEqual(result["already_unreachable_turns"], 1)
        self.assertEqual(result["missing_reachability_turns"], 0)
        self.assertEqual(result["keeps_goal_reachable"], 1)
        self.assertEqual(result["action_environment_shortest"], 1)

    def test_replay_preserves_executed_history_and_scores_last_choice(self):
        environment = task_environment(*line_graph())
        record = dict(trajectory_id="tiny", start=0, goal=2, sample_seed=7, shortest_moves=2)
        session = Mock()
        session.ask.return_value = ("<action>1</action>", dict(context_exhausted=False))
        case = closed_loop(session, record, "", environment,
                           dict(maximum_demonstration_actions=1, data={}), 0)
        case.update(mode="rollout", trajectory_id="tiny", variant=0)
        self.assertEqual(case["actual_path"], ["0", "1"])
        self.assertFalse(case["reached_goal"])
        with patch("experiments.flamingo_map_reader.src.trajectory_eval.TaskEnvironment",
                   return_value=environment):
            self.assertEqual(complete_rollout_reachability([case], dict(config={}, records=[record])), 1)
        self.assertEqual(audit([case])["keeps_goal_reachable"], 2)
        self.assertEqual(case["actual_path"], ["0", "1"])
        self.assertFalse(case["reached_goal"])

    def test_irregular_saves_and_extended_final_keep_their_recorded_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = {"checkpoint-128": dict(step=128, epoch=128 / 49000),
                      "checkpoint-196000": dict(step=196000, epoch=4.0),
                      "final": dict(step=245000, epoch=5.0)}
            for name, state in states.items():
                directory = root / "blocks/training/models" / name
                directory.mkdir(parents=True)
                (directory / "evaluation_ready.json").write_text(json.dumps(state))
            (root / "blocks/training/models/checkpoint-245000").mkdir()
            self.assertEqual(checkpoint_epochs(root), {"blocks": states})


if __name__ == "__main__":
    unittest.main()
