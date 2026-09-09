import importlib.util
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

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
        text = '{"path":[{"from":0,"to":1}],"final_node":1}'
        reasoning, answer, verdict = runner.verdict_for(self.case, text, 'stop')
        self.assertEqual(answer, '')
        self.assertFalse(verdict['pass'])
        self.assertEqual(reasoning, text)

    def test_final_answer_uses_existing_judge(self):
        answer = json.dumps({'path': [{'from': 0, 'to': 1}], 'final_node': 1})
        _, extracted, verdict = runner.verdict_for(self.case, '<think>{}\n</think>\n' + answer, 'stop')
        self.assertEqual(extracted, answer)
        self.assertEqual(verdict, runner.sol.evaluate(self.case, answer))
        self.assertTrue(verdict['pass'])

    def test_truncation_is_failure_even_with_final_json(self):
        _, _, verdict = runner.verdict_for(self.case, '</think>{"path":[{"from":0,"to":1}],"final_node":1}', 'length')
        self.assertFalse(verdict['pass'])
        self.assertEqual(verdict['failure_type'], 'budget_truncated')
        self.assertTrue(verdict['partial_verdict']['pass'])

    def test_abort_is_infrastructure_error(self):
        with self.assertRaises(ValueError):
            runner.verdict_for(self.case, 'unfinished', 'abort')

    def test_full_batch_summary_and_resume_without_resampling(self):
        from prepare import prepare
        suite = prepare()
        outputs = []
        for case in suite['cases']:
            if case['condition'].startswith('blocks'):
                task = runner.sol.TASKS[case['condition']]
                mask = task.from_grid(case['grid'])
                actions = copy.deepcopy(case['construction_reference'])
                for action in actions:
                    mask = task.apply(mask, action)
                    action['board_after'] = task.to_rows(mask)
                answer = {'actions': actions, 'final_status': 'solved'}
            else:
                answer = {'path': case['reference']['path'], 'final_node': case['goal']}
            outputs.append('<think>test reasoning</think>' + json.dumps(answer))

        class Tokenizer:
            chat_template = 'enable_thinking'
            def __init__(self):
                self.index = -1
            def apply_chat_template(self, *args, **kwargs):
                self.index += 1
                return '<|im_start|>assistant\n'
            def encode(self, *args, **kwargs):
                return [self.index]

        fake = types.ModuleType('transformers')
        fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
        def respond(request, **kwargs):
            index = json.loads(request.data)['input_ids'][0]
            return io.StringIO(json.dumps({'text': outputs[index], 'meta_info': {
                'finish_reason': {'type': 'stop'}, 'prompt_tokens': 1, 'completion_tokens': 100}}))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            suite_path = directory / 'suite.json'
            suite_path.write_text(json.dumps(suite))
            argv = ['run.py', '--config', str(ROOT / 'experiments/qwen_path_blocks/configs/thinking.json'),
                    '--suite', str(suite_path), '--model-id', 'Qwen/Qwen3-4B', '--model-path', 'test-model',
                    '--endpoints', 'http://test', '--out', str(directory / 'out')]
            with patch.dict(sys.modules, {'transformers': fake}), patch('importlib.metadata.version', return_value='test'), \
                    patch.object(runner.urllib.request, 'urlopen', side_effect=respond) as calls, \
                    contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', argv):
                    runner.main()
                self.assertEqual(calls.call_count, 384)
                summary = json.loads((directory / 'out/summary.json').read_text())
                self.assertEqual(sum(s['successes'] for s in summary['conditions'].values()), 384)
                with patch.object(sys, 'argv', argv + ['--resume']):
                    runner.main()
                self.assertEqual(calls.call_count, 384)


if __name__ == '__main__':
    unittest.main()
