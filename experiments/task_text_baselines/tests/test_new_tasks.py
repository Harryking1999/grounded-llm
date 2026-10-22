"""New-task plan formatting must preserve the exact environments and scoring contract."""
import json
import unittest
from grounded_llm.tasks.text import judge, prompt, ROOT, section_text
from experiments.task_text_baselines.src.prepare_new_tasks import cases_from_run


class NewTaskTextTest(unittest.TestCase):
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
