"""A small closed loop checks the graph evaluator's action and done boundaries."""

import unittest

import torch

from experiments.flamingo_map_reader.src.evaluate_graph import rollout
from test_graph import line_graph
from test_timeline import ByteChatTemplate


class ByteTokenizer(ByteChatTemplate):
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids.tolist() if hasattr(ids, "tolist") else ids).decode("utf-8")


class ScriptedReader:
    def __init__(self):
        self.answers = iter(("<action>1</action>", "<done/>"))

    def generate(self, timeline, input_ids, **kwargs):
        answer = next(self.answers).encode("utf-8")
        return torch.cat((input_ids, torch.tensor([list(answer)], device=input_ids.device)), dim=1)


class GraphEvalTest(unittest.TestCase):
    def test_actual_step_then_model_terminal_answer(self):
        environment, qmap = line_graph()
        case = {"id": "case_0", "start": 0, "goal": 1,
                "node_order": [0, 1, 2],
                "neighbors": {"0": [1], "1": [0, 2], "2": [1]},
                "reference": {"length": 1}}
        config = {"seed": 1, "maximum_demonstration_actions": 3,
                  "maximum_sequence_tokens": 2000,
                  "evaluation": {"action_max_new_tokens": 50,
                                 "terminal_max_new_tokens": 50}}
        result = rollout(ScriptedReader(), ByteTokenizer(), environment, qmap,
                         case, config, torch.device("cpu"))
        self.assertEqual(result["path"], [0, 1])
        self.assertTrue(result["success"])
        self.assertTrue(result["shortest_success"])
        self.assertEqual(len(result["trace"]), 2)


if __name__ == "__main__":
    unittest.main()
