import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('qwen_runner', ROOT / 'experiments/qwen_path_blocks/src/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = {'id': 'path_test', 'condition': 'path_dag', 'start': 0, 'goal': 1,
                    'neighbors': {'0': [1], '1': []}, 'reference': {'length': 1, 'path': [1]}}

    def test_reasoning_json_never_becomes_answer(self):
        text = '{"path":[1],"final_node":1}'
        reasoning, answer, verdict = runner.verdict_for(self.case, text, 'stop')
        self.assertEqual(answer, '')
        self.assertFalse(verdict['pass'])
        self.assertEqual(reasoning, text)

    def test_final_answer_uses_existing_judge(self):
        answer = json.dumps({'path': [1], 'final_node': 1})
        _, extracted, verdict = runner.verdict_for(self.case, '{}\n</think>\n' + answer, 'stop')
        self.assertEqual(extracted, answer)
        self.assertEqual(verdict, runner.sol.evaluate(self.case, answer))

    def test_truncation_is_failure_even_with_final_json(self):
        _, _, verdict = runner.verdict_for(self.case, '</think>{"path":[1]}', 'length')
        self.assertFalse(verdict['pass'])
        self.assertEqual(verdict['failure_type'], 'budget_truncated')

    def test_abort_is_infrastructure_error(self):
        with self.assertRaises(ValueError):
            runner.verdict_for(self.case, 'unfinished', 'abort')


if __name__ == '__main__':
    unittest.main()
