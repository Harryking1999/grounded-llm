import json
import random
import unittest

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.task_text_baselines.src import cube2, hanoi, puzzle8
from experiments.task_text_baselines.src.protocol import judge, prompt
from experiments.task_text_baselines.src.run import extract_text, pending
from experiments.task_text_baselines.src.qwen import final_text, round_robin


class TaskRulesTest(unittest.TestCase):
    def test_blank_directions_edges_and_inverses(self):
        start = (3,1,2,4,0,5,6,7,8)
        self.assertEqual(puzzle8.step(start, "Left"), (3,1,2,0,4,5,6,7,8))
        for action, inverse in (("Left","Right"),("Up","Down")):
            self.assertEqual(puzzle8.step(puzzle8.step(start,action),inverse), start)
        with self.assertRaises(ValueError):
            puzzle8.step(tuple(range(9)), "Up")
        with self.assertRaises(ValueError):
            puzzle8.validate((0,0,1,2,3,4,5,6,7))

    def test_eight_disk_single_goal_distances(self):
        table = hanoi.distances((2,) * 8)
        self.assertEqual(len(table), 3**8)
        self.assertEqual(table[(0,) * 8], 255)
        state = (0,) * 8
        actions = hanoi.solve(state,table)
        for action in actions:
            state = hanoi.step(state,action)
        self.assertEqual(state,(2,) * 8)

    def test_hanoi_matches_existing_environment(self):
        old = Hanoi(3)
        for state_id, state in enumerate(old.positions):
            actual = {old.state_id(child) for _,child in hanoi.successors(tuple(map(int,state)))}
            expected = {int(child) for child in old.successors([state_id])[0] if child >= 0}
            self.assertEqual(actual,expected)
        with self.assertRaises(ValueError):
            hanoi.step((0,0,0), {"disk_id":2,"from":"A","to":"B"})
        with self.assertRaises(ValueError):
            hanoi.step((1,0,0), {"disk_id":2,"from":"A","to":"B"})

    def test_cube_conversion_roundtrip_and_known_move(self):
        env = cube2.environment()
        for state in random.Random(4).sample(range(env.size),64):
            self.assertEqual(cube2.from_grids(cube2.grids(state)),state)
        right = cube2.grids(env.step(0,"R"))
        self.assertEqual(right["Upper"],[0,2,0,2])
        self.assertEqual(right["Back"],[0,5,0,5])
        with self.assertRaises(ValueError):
            cube2.step(0,"R2")

    def test_judge_stops_first_illegal_and_checks_final_state(self):
        case={"task":"puzzle8","initial":[1,0,2,3,4,5,6,7,8],"goal":list(range(9)),"reference":{"length":1}}
        good=judge(case,json.dumps({"actions":["Left"],"final_status":"solved"}))
        self.assertTrue(good["contract_pass"])
        wrong=judge(case,json.dumps({"actions":["Right"],"final_status":"solved"}))
        self.assertFalse(wrong["solved"])
        illegal=judge(case,json.dumps({"actions":["Up","Left"],"final_status":"solved"}))
        self.assertEqual(illegal["first_illegal_index"],0)
        self.assertEqual(illegal["executed"],0)
        fenced=judge(case,'```json\n'+json.dumps({"actions":["Left"],"final_status":"solved"})+'\n```')
        self.assertTrue(fenced["solved"])
        self.assertFalse(fenced["contract_pass"])

    def test_prompt_uses_current_rules_not_oracle(self):
        case={"task":"puzzle8","initial":[1,0,2,3,4,5,6,7,8],"goal":list(range(9)),"reference":{"length":1,"actions":["Left"]}}
        text=prompt(case)
        self.assertIn('one blank, represented by 0',text)
        self.assertIn('1 0 2\n3 4 5\n6 7 8',text)
        self.assertNotIn('After Left',text)
        self.assertNotIn('reference',text)
        self.assertNotIn('Rows are shown',text)
        self.assertIn('There is no intermediate feedback',text)

    def test_final_channel_has_priority(self):
        response={"output":[{"type":"message","channel":"commentary","content":[{"type":"output_text","text":"working"}]},
                            {"type":"message","channel":"final","content":[{"type":"output_text","text":"answer"}]}]}
        self.assertEqual(extract_text(response),"answer")

    def test_resume_only_retries_service_failures_and_prioritizes_them(self):
        cases = [{"id":name} for name in ("ok","wrong","truncated","network","new","exhausted")]
        latest = {name:{"attempt":1} for name in ("ok","wrong","truncated")}
        latest.update(network={"attempt":1,"service_error":"URLError"},
                      exhausted={"attempt":6,"service_error":"URLError"})
        rows = pending(cases,latest,{"max_service_retries":5},42)
        self.assertEqual(rows.pop()["id"],"network")
        self.assertEqual([row["id"] for row in rows],["new"])

    def test_qwen_keeps_final_answer_and_stable_case_indices(self):
        class Tokenizer:
            def convert_tokens_to_ids(self, value):
                return 9
            def decode(self, values, skip_special_tokens):
                return ",".join(map(str,values))
        self.assertEqual(final_text([1,2,9,3,4],Tokenizer()),("3,4",3))
        cases = [{"task":task} for task in ("a","a","b","b","c")]
        self.assertEqual([index for index,_ in round_robin(cases)],[0,2,4,1,3])


if __name__ == '__main__':
    unittest.main()
