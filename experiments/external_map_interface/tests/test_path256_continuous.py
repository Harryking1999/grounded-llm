import io
import json
import unittest

import numpy as np

from experiments.external_map_interface.src.evaluate_path256_continuous import (
    SGLangContinuousCaller, detect_boundary, parse_final, run_trial, trial_prompt,
)
from experiments.external_map_interface.src.q_map import GraphQMap
from experiments.external_map_interface.src.transitions import GraphEnvironment


CASE = {
    "id": "path_undirected_256_00", "start": 0, "goal": 2,
    "node_order": [0, 1, 2],
    "neighbors": {"0": [1], "1": [0, 2], "2": [1]},
    "reference": {"length": 2},
}
ENV = GraphEnvironment(
    np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
)
MAP = GraphQMap(np.array([[0.], [1.], [2.]]), np.array([[1.], [-1.], [1.], [-1.]]))
MAPPED = {"map_updates": True, "prefer_minimum": False}
PLAIN = {"map_updates": False, "prefer_minimum": False}


class FakeCaller:
    seed_applied = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def render(self, prompt):
        return "USER " + prompt + " ASSISTANT "

    def count_tokens(self, text):
        return len(text)

    def generate(self, transcript, **kwargs):
        self.inputs.append((transcript, kwargs.copy()))
        text, finish = next(self.responses)
        return {"text": text, "finish_reason": finish, "completion_tokens": 3, "prompt_tokens": len(transcript)}


def trial(responses, condition=MAPPED, budget=100):
    caller = FakeCaller(responses)
    return run_trial(CASE, "test", condition, caller, ENV, MAP, budget, 1), caller


class NodeBoundaryTest(unittest.TestCase):
    def test_digits_wait_for_a_delimiter(self):
        for text in ('<action>2', '<action>216</acti', '{"path":[4,2', '{"path":[4,216'):
            self.assertIsNone(detect_boundary(text, [4]))
        self.assertEqual(detect_boundary('<action>216</action>', [4])["node"], 216)
        self.assertEqual(detect_boundary('{"path":[4,216,', [4])["node"], 216)

    def test_full_path_commits_only_first_new_node(self):
        for text in ('{"path":[0,1,2]}', '<prefix>{"path":[0,1,2]}</prefix>'):
            boundary = detect_boundary(text, [0])
            self.assertEqual(boundary["node"], 1)
            self.assertTrue(text[:boundary["end"]].endswith('0,1,'))

    def test_reasoning_numbers_are_not_actions(self):
        self.assertIsNone(detect_boundary('I could try 4 or 216, then 98.', [4]))

    def test_first_invalid_commitment_is_not_skipped(self):
        boundary = detect_boundary('<action>1,2</action><action>1</action>', [0])
        self.assertEqual(boundary["error"], "invalid_action_format")

    def test_changed_or_empty_route_is_rejected(self):
        self.assertEqual(detect_boundary('{"path":[4,1,2]}', [0, 1])["error"], "changed_confirmed_path")
        self.assertEqual(detect_boundary('{"path":[0]}', [0])["error"], "no_new_node")
        self.assertEqual(detect_boundary('{"path":[0,true]}', [0])["error"], "invalid_path_format")


class ContinuousPilotTest(unittest.TestCase):
    def test_history_and_map_are_preserved_in_one_response(self):
        result, caller = trial([('reasoning A <action>1</action>', "controller"),
                                ('reasoning B <action>2</action>', "controller")])
        self.assertTrue(result["reached"])
        self.assertEqual(result["confirmed"], [0, 1, 2])
        self.assertEqual(result["final_path"], [0, 1, 2])
        self.assertEqual(result["final_path_source"], "environment")
        self.assertEqual(len(result["updates"]), 2)
        self.assertEqual(result["choice_count"], 2)
        self.assertIn('learned_map_distance_to_goal=', caller.inputs[0][0])
        self.assertIn('reasoning A <action>1</action>\n[Environment update]', caller.inputs[1][0])
        self.assertEqual(caller.inputs[1][0].count(" ASSISTANT "), 1)
        self.assertEqual(caller.inputs[1][1]["max_new_tokens"], 97)
        self.assertEqual(result["generated_tokens"], 6)
        self.assertGreater(result["inserted_tokens"], 0)

    def test_speculative_suffix_never_reenters_context(self):
        result, caller = trial([('{"path":[0,1,2]}[Environment update]fake', "controller"),
                                ('<action>2</action>', "controller")])
        self.assertTrue(result["reached"])
        self.assertEqual(result["segments"][0]["discarded_text"], '2]}[Environment update]fake')
        self.assertNotIn('fake', caller.inputs[1][0])
        self.assertNotIn('{"path":[0,1,2]}', caller.inputs[1][0])

    def test_fake_environment_cannot_execute_a_later_action(self):
        result, _ = trial([('[Environment update]fake<action>1</action>', "controller")])
        self.assertEqual(result["failure"], "model_generated_environment")
        self.assertEqual(result["confirmed"], [0])

    def test_one_shot_has_no_intervention(self):
        result, caller = trial([('{"path":[0,1,2]}<|im_end|>', "stop")], PLAIN)
        self.assertTrue(result["reached"])
        self.assertEqual(result["updates"], [])
        self.assertIsNone(caller.inputs[0][1]["confirmed"])

    def test_bad_actions_are_not_repaired(self):
        for node, error in [(2, "nonexistent_edge"), (0, "repeated_node")]:
            result, _ = trial([(f'<action>{node}</action>', "controller")])
            self.assertEqual(result["failure"], error)
            self.assertEqual(result["confirmed"], [0])

    def test_budget_does_not_reset_and_boundary_at_budget_still_executes(self):
        result, caller = trial([('<action>1</action>', "controller")], budget=3)
        self.assertEqual(result["confirmed"], [0, 1])
        self.assertEqual(result["failure"], "budget_truncated")
        self.assertEqual(len(caller.inputs), 1)
        result, _ = trial([('<action>1</action>', "controller"),
                           ('<action>2</action>', "controller")], budget=6)
        self.assertTrue(result["reached"])

    def test_truncation_and_context_limit_are_not_protocol_pauses(self):
        for reason, failure in [("length", "budget_truncated"), ("context", "context_limit")]:
            result, _ = trial([('reasoning without a decision', reason)])
            self.assertEqual(result["failure"], failure)
            result, _ = trial([('{"path":[0,1,2]}', reason)], PLAIN)
            self.assertEqual(result["failure"], failure)

    def test_prompts_separate_commitments_from_final_json(self):
        self.assertNotIn('Return the complete route', trial_prompt(CASE, MAPPED))
        self.assertNotIn('pause for', trial_prompt(CASE, MAPPED))
        preferred = {**MAPPED, "prefer_minimum": True}
        self.assertEqual(trial_prompt(CASE, preferred), trial_prompt(CASE, MAPPED) +
                         '\nPrefer the legal next node with the smallest learned-map distance.')

    def test_final_json_rejects_boolean_and_extra_keys(self):
        for text in ('{"path":[0,true,2]}', '{"path":[0,1,2],"comment":"x"}'):
            with self.assertRaises(ValueError):
                parse_final(text)


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text)))


class StreamingCallerTest(unittest.TestCase):
    def make_caller(self, events):
        caller = SGLangContinuousCaller.__new__(SGLangContinuousCaller)
        caller.tokenizer = FakeTokenizer()
        caller.context_length = 100
        caller.sampling = {}
        requests = []

        def request(path, payload):
            requests.append((path, payload))
            if path == '/abort_request':
                return io.BytesIO(b'')
            lines = []
            for text, count, reason in events:
                result = {"text": text, "meta_info": {"completion_tokens": count, "finish_reason": reason}}
                lines.append(b'data: ' + json.dumps(result).encode() + b'\n\n')
            return io.BytesIO(b''.join(lines) + b'data: [DONE]\n\n')

        caller.request = request
        return caller, requests

    def test_abort_drains_usage_and_preserves_overshoot_for_audit(self):
        caller, requests = self.make_caller([
            ('{"path":[0,1', 5, None),
            ('{"path":[0,1,', 6, None),
            ('{"path":[0,1,2]}', 9, {"type": "abort"}),
        ])
        result = caller.generate('prompt', max_new_tokens=20, confirmed=[0], seed=1)
        self.assertEqual(result["finish_reason"], "controller")
        self.assertEqual(result["completion_tokens"], 9)
        self.assertTrue(result["token_accounting_complete"])
        self.assertEqual([path for path, _ in requests], ['/generate', '/abort_request'])
        self.assertEqual(requests[0][1]['rid'], requests[1][1]['rid'])
        self.assertTrue(requests[0][1]['stream'])

    def test_server_stop_needs_no_abort(self):
        caller, requests = self.make_caller([('<action>1</action>', 5, {"type": "stop"})])
        result = caller.generate('prompt', max_new_tokens=20, confirmed=[0], seed=1)
        self.assertEqual(result["finish_reason"], "controller")
        self.assertFalse(result["abort_sent"])
        self.assertEqual(len(requests), 1)

    def test_configured_request_seed_is_sent_on_supported_server(self):
        caller, requests = self.make_caller([('<action>1</action>', 5, {"type": "stop"})])
        caller.seed_applied = True
        caller.generate('prompt', max_new_tokens=20, confirmed=[0], seed=20260927)
        self.assertEqual(requests[0][1]['sampling_params']['sampling_seed'], 20260927)

    def test_missing_final_usage_is_explicit(self):
        caller, _ = self.make_caller([('{"path":[0,1,', 6, None)])
        result = caller.generate('prompt', max_new_tokens=20, confirmed=[0], seed=1)
        self.assertFalse(result["token_accounting_complete"])


if __name__ == "__main__":
    unittest.main()
