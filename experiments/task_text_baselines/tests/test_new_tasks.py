"""New-task plan formatting must preserve the exact environments and scoring contract."""
import json
import unittest
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from grounded_llm.tasks.text import judge, prompt, ROOT, section_text
from experiments.task_text_baselines.src.prepare_new_tasks import cases_from_run
from grounded_llm.tasks.bridge.environment import Bridge
from grounded_llm.tasks.bridge.solve import optimal_plan
from grounded_llm.tasks.game24.environment import Game24
from grounded_llm.tasks.game24.solve import solve as solve24
from grounded_llm.tasks.sokoban.environment import Sokoban
from grounded_llm.tasks.sokoban.solve import solve as solve_boxes
from experiments.task_text_baselines.src.escalate import main as escalate_main


class NewTaskTextTest(unittest.TestCase):
    def test_bridge_recurrence_matches_exact_small_graphs(self):
        rng = random.Random(7)
        for n in range(2, 8):
            for _ in range(5):
                times = [rng.randint(1, 20) for _ in range(n)]
                actions, cost = optimal_plan(times)
                env = Bridge(times)
                self.assertEqual(env.graph()['distances'][0], cost)
                self.assertTrue(env.judge(actions)['optimal'])

    def test_direct_bridge_replay_matches_enumerated_transitions(self):
        env = Bridge([3, 1, 8, 2, 5])
        for state in range(2 ** (env.n + 1)):
            for action, target, cost in env.transitions(state):
                self.assertEqual(env.apply(state, env.actions[action]), (target, cost))

    def test_push_solver_witness_and_unknown_move_optimum(self):
        board = ['######', '# .  #', '# $$ #', '# @. #', '#    #', '######']
        env = Sokoban(board)
        actions, _ = solve_boxes(env)
        self.assertIsNotNone(actions)
        self.assertTrue(env.judge(actions)['success'])
        case = {'task': 'sokoban', 'initial': board, 'action_limit': 512, 'reference': {'cost': None}}
        verdict = judge(case, json.dumps({'actions': actions, 'final_status': 'solved'}))
        self.assertTrue(verdict['solved'])
        self.assertIsNone(verdict['optimal'])

    def test_fraction_required_probe_has_a_valid_witness(self):
        state = Game24.canonical([4, 4, 7, 7])
        self.assertIsNone(solve24(state, True))
        actions = solve24(state)
        self.assertTrue(Game24().judge([4, 4, 7, 7], actions)['success'])

    def test_first_error_stops_and_resume_preserves_it(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            config = {'policy': 'first_valid_model_error_per_task', 'study': 'adaptive_complete_plan_difficulty',
                      'seed': 1, 'api_config': 'api.json', 'concurrency': 1, 'cases_per_level': 3,
                      'confirmation_cases': 3, 'initial_levels': {'bridge': 0},
                      'tasks': {'bridge': {'people_counts': [8, 12]}}}
            api = {'api': {'model': 'mock', 'reasoning_effort': 'high', 'max_service_retries': 1,
                           'max_total_service_retries': 2, 'service_retry_delay_seconds': 0}}
            (root/'config.json').write_text(json.dumps(config))
            (root/'api.json').write_text(json.dumps(api))
            cases = [{'id': f'bridge_level0_{i:02d}', 'task': 'bridge', 'level': 0, 'level_value': 8,
                      'initial': 0, 'goal': 'all_right', 'times': [1+i, 2], 'stratum': 'test'} for i in range(3)]
            def fail(case, api, key, attempt):
                return {'case_id': case['id'], 'task': 'bridge', 'stratum': 'test', 'attempt': attempt,
                        'verdict': {'solved': False, 'failure': 'illegal_action'}}
            with patch('experiments.task_text_baselines.src.escalate.subprocess.check_output', return_value='test-source'), patch('experiments.task_text_baselines.src.escalate.ROOT', root), patch.dict('os.environ', {'GND_API_KEY': 'mock'}), patch('sys.argv', ['run', '--config', str(root/'config.json'), '--out', str(root/'out')]), patch('experiments.task_text_baselines.src.escalate.prepare_level', return_value=cases), patch('experiments.task_text_baselines.src.escalate.call', side_effect=fail) as request, patch('experiments.task_text_baselines.src.escalate.build', return_value=({}, '')), patch('builtins.print'):
                escalate_main()
                escalate_main()
            self.assertEqual(request.call_count, 1)
            state = json.loads((root/'out/progress.json').read_text())
            self.assertEqual(state['tasks']['bridge']['status'], 'model_error_found')

    def test_bridge_cost_is_time_not_crossing_count(self):
        case = {'task': 'bridge', 'times': [1, 2, 5, 10], 'initial': 0,
                'action_limit': 128, 'reference': {'cost': 17}}
        actions = [[0, 1], [0], [2, 3], [1], [0, 1]]
        result = judge(case, json.dumps({'actions': actions, 'final_status': 'solved'}))
        self.assertTrue(result['optimal'])
        self.assertEqual(result['cost'], 17)
        case['reference']['cost'] = 16
        self.assertTrue(judge(case, json.dumps({'actions': actions, 'final_status': 'solved'}))['solved'])
        self.assertFalse(judge(case, json.dumps({'actions': actions, 'final_status': 'solved'}))['optimal'])
        for action in [[True], [0, 0], ['0'], [99]]:
            self.assertEqual(judge(case, json.dumps({'actions': [action]}))['failure'], 'illegal_action')

    def test_sokoban_goal_is_boxes_not_player(self):
        case = {'task': 'sokoban', 'initial': ['#####', '#@$.#', '#####'],
                'action_limit': 128, 'reference': {'cost': 1}}
        result = judge(case, '{"actions":["R"],"final_status":"solved"}')
        self.assertTrue(result['contract_pass'])
        result = judge(case, '{"actions":["R"],"final_status":"unsolved"}')
        self.assertTrue(result['solved'])
        self.assertFalse(result['contract_pass'])
        self.assertEqual(judge(case, '{"actions":["U"]}')['failure'], 'illegal_action')

    def test_game24_current_sorted_slots_and_exact_fractions(self):
        case = {'task': 'game24', 'initial': [3, 3, 8, 8], 'action_limit': 3, 'reference': {'cost': 3}}
        result = judge(case, json.dumps({'actions': [[0, 'r/', 2], [0, 'r-', 1], [0, 'r/', 1]], 'final_status': 'solved'}))
        self.assertTrue(result['optimal'])
        self.assertEqual(judge(case, '{"actions":[[0,"/",0]]}')['failure'], 'illegal_action')

    def test_all_task_wording_has_renderable_common_sections(self):
        for task in ('blocks', 'path', 'hanoi', 'cube2', 'puzzle8', 'bridge', 'sokoban', 'game24'):
            self.assertTrue(section_text(task, '共享任务描述与规则'))
            self.assertTrue(section_text(task, 'SOTA 组／32B 组'))
            text = (ROOT / 'tasks' / task / 'prompts/README.md').read_text(encoding='utf-8')
            self.assertIn('[Environment update]', text)
            self.assertIn('<done/>', text)
            self.assertIn('Reason:', text)


if __name__ == '__main__': unittest.main()
