import json
import random
import unittest
from unittest.mock import patch
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.task_text_baselines.src import cube2, cube3, hanoi, puzzle8, puzzle15, sliding
from experiments.task_text_baselines.src.difficulty import graph_distances, prepare_level
from experiments.task_text_baselines.src.escalate import decision, main as escalate_main
from experiments.task_text_baselines.src.protocol import ROOT
from experiments.task_text_baselines.src.protocol import judge, prompt
from experiments.task_text_baselines.src.run import call, extract_text, pending, summary
from experiments.task_text_baselines.src.qwen import final_text, round_robin
from experiments.task_text_baselines.src.prepare_difficulty import reuse_answers


class TaskRulesTest(unittest.TestCase):
    def test_new_queue_rejects_five_case_contracts(self):
        config = {'cases_per_level':5,'confirmation_cases':20}
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/'config.json'
            path.write_text(json.dumps(config))
            with patch('sys.argv',['escalate','--config',str(path),'--out',temporary]):
                with self.assertRaisesRegex(ValueError,'at least twenty'):
                    escalate_main()

    def test_expanded_suite_reuses_wrong_answers_without_resampling(self):
        case = {'id':'old','task':'path','initial':0,'goal':1,'prompt':'frozen',
                'neighbors':{'0':[1],'1':[0]},'reference':{'length':1}}
        raw = json.dumps({'actions':[],'final_status':'unsolved'})
        record = {'case_id':'old','attempt':1,'raw_output':raw,'verdict':judge(case,raw)}
        with TemporaryDirectory() as temporary:
            prior,out = Path(temporary)/'prior',Path(temporary)/'out'
            (prior/'records').mkdir(parents=True)
            (out/'records').mkdir(parents=True)
            (prior/'progress.json').write_text(json.dumps({'status':'stopped','inflight':[]}))
            (prior/'suite.json').write_text(json.dumps({'source_commit':'prior','cases':[case]}))
            (prior/'records/old_attempt1.json').write_text(json.dumps(record))
            reused = reuse_answers([case | {'id':'expanded'}],prior,out)
            self.assertEqual(len(reused),1)
            saved = json.loads((out/'records/expanded_attempt1.json').read_text())
            self.assertFalse(saved['verdict']['solved'])
            self.assertEqual(saved['reused_from']['case_id'],'old')
            (prior/'progress.json').write_text(json.dumps({'status':'running','inflight':['old']}))
            with self.assertRaises(RuntimeError):
                reuse_answers([case],prior,out)

    def test_new_board_conditions_render_and_replay_all_twenty(self):
        config = json.loads((ROOT/'experiments/task_text_baselines/configs/difficulty_20.json').read_text())
        for task in ('blocks','puzzle24'):
            cases = prepare_level(task,0,config['tasks'][task],config)
            self.assertEqual(len(cases),20)
            self.assertEqual(len({str(case['initial']) for case in cases}),20)
            for case in cases:
                self.assertTrue(judge(case,json.dumps({'actions':case['reference']['actions'],'final_status':'solved'}))['solved'])
                if task == 'blocks':
                    self.assertEqual(case['shape_options']['grid_size'],15)
                    self.assertEqual(case['generation_placements'],20)
                    self.assertTrue(any(action['shape_id'] >= 6 for action in case['reference']['actions']))
                    board = case['prompt'].split('Initial board:\n')[1].split('\n\n')[0].splitlines()
                    self.assertEqual([len(row) for row in board],[15]*15)
                    self.assertIn('15x15',case['prompt'])
                    self.assertIn('6=011/110',case['prompt'])
                else:
                    self.assertIn('5x5',case['prompt'])
                    self.assertIn('1 through 24',case['prompt'])
                    self.assertNotIn('Action examples',case['prompt'])
                    self.assertGreaterEqual(case['manhattan_lower_bound'],60)
        state = tuple(range(25))
        self.assertEqual(sliding.step(sliding.step(state,'Right'),'Left'),state)
        with self.assertRaises(ValueError):
            sliding.step(state,'Up')
        last_blank = tuple(range(1,25))+(0,)
        for action in ('Down','Right'):
            with self.assertRaises(ValueError):
                sliding.step(last_blank,action)

    def test_exhausted_service_attempts_remain_incomplete_without_new_calls(self):
        config_path = ROOT/'experiments/task_text_baselines/configs/path_distance_stress.json'
        config = json.loads(config_path.read_text())
        config.pop('retired',None)
        config['cases_per_level'] = config['confirmation_cases'] = 20
        case = {'id':'pending','task':'path','level':0,'phase':'screen','stratum':'long'}
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config_path = folder/'config.json'
            config_path.write_text(json.dumps(config),encoding='utf-8')
            (folder/'records').mkdir()
            cases = [case]+[case | {'id':f'untouched_{i}'} for i in range(19)]
            values = {'suite.json':{'config':config,'source_commit':'test','cases':cases},
                      'records/pending_attempt11.json':{'case_id':'pending','task':'path','stratum':'long',
                          'attempt':11,'service_error':'HTTPError','verdict':{'solved':False,'failure':'service_error'}}}
            for name,value in values.items():
                (folder/name).write_text(json.dumps(value),encoding='utf-8')
            with patch('sys.argv',['escalate','--config',str(config_path),'--out',str(folder)]), patch.dict(
                    'os.environ',{'GND_API_KEY':'test-key'}), patch(
                    'experiments.task_text_baselines.src.escalate.subprocess.check_output',return_value='test'), patch(
                    'experiments.task_text_baselines.src.escalate.call') as api_call:
                escalate_main()
            api_call.assert_not_called()
            result = json.loads((folder/'progress.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'service_incomplete')
            self.assertEqual(result['tasks']['path']['status'],'service_blocked')

    def test_path_distance_stress_is_connected_undirected_and_long(self):
        config = json.loads((ROOT/'experiments/task_text_baselines/configs/path_distance_stress.json').read_text())
        config['cases_per_level'] = 2
        for case in prepare_level('path',0,config['tasks']['path'],config):
            graph = case['neighbors']
            self.assertEqual(sum(map(len,graph.values())),2*(len(graph)-1))
            integer_graph = {int(node):neighbors for node,neighbors in graph.items()}
            self.assertEqual(len(graph_distances(integer_graph,0)),len(graph))
            for source,neighbors in graph.items():
                for target in neighbors:
                    self.assertIn(int(source),graph[str(target)])
            self.assertGreaterEqual(case['reference']['length'],64)
            self.assertNotIn('reference',case['prompt'])

    def test_service_retry_waits_without_changing_the_request(self):
        case = {'id':'retry','task':'path','stratum':'hard','prompt':'frozen'}
        api = {'model':'requested','reasoning_effort':'high','max_output_tokens':32768,
               'base_url':'https://example.invalid/v1','timeout_seconds':1,
               'service_retry_delay_seconds':30}
        with patch('experiments.task_text_baselines.src.run.time.sleep') as sleep, patch(
                'experiments.task_text_baselines.src.run.urllib.request.urlopen',
                side_effect=urllib.error.URLError('temporary')) as open_request:
            record = call(case,api,'test-key',2)
        sleep.assert_called_once_with(30)
        body = json.loads(open_request.call_args.args[0].data)
        self.assertEqual(body['input'],'frozen')
        self.assertEqual(body['max_output_tokens'],32768)
        self.assertEqual(record['service_error'],'URLError')

    def test_stratum_separates_service_errors_from_valid_failure(self):
        cases = [{'id':name,'task':'path'} for name in ('ok','wrong','network')]
        records = [{'task':'path','stratum':'hard','verdict':{'solved':True}},
                   {'task':'path','stratum':'hard','verdict':{'solved':False,'failure':'illegal_action'}},
                   {'task':'path','stratum':'hard','service_error':'HTTPError',
                    'verdict':{'solved':False,'failure':'service_error'}}]
        row = summary(records,cases)['path']['strata']['hard']
        self.assertEqual(row,{'finished':3,'valid_responses':2,'service_errors':1,'solved':1})

    def test_difficulty_requires_independent_confirmation(self):
        config = {'maximum_solved_fraction':0.5}
        def rows(solved,total):
            return [{'verdict':{'solved':i<solved}} for i in range(total)]
        self.assertEqual(decision(rows(4,5),'screen',config),'advance')
        self.assertEqual(decision(rows(2,5),'screen',config),'confirm')
        self.assertEqual(decision(rows(5,10),'confirm',config),'difficulty_confirmed')
        self.assertEqual(decision(rows(6,10),'confirm',config),'advance')

    def test_puzzle15_distance_stress_has_provable_floor_and_valid_witness(self):
        config = json.loads((ROOT/'experiments/task_text_baselines/configs/puzzle15_distance_stress.json').read_text())
        config['cases_per_level'] = 2
        for case in prepare_level('puzzle15',0,config['tasks']['puzzle15'],config):
            self.assertGreaterEqual(case['manhattan_lower_bound'],40)
            self.assertLessEqual(case['manhattan_lower_bound'],case['reference']['upper_bound'])
            self.assertNotIn('Manhattan',case['prompt'])

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

    def test_cube3_corner_frames_match_authoritative_cube2(self):
        state2,state = 0,cube3.solved(size=2)
        rng = random.Random(41)
        for _ in range(60):
            action = rng.choice(("U","U'","R","R'","F","F'"))
            state2,state = cube2.step(state2,action),cube3.step(state,action,size=2)
            expected = cube2.grids(state2)
            self.assertEqual(list(state),[v for face in cube3.FACES for v in expected[face]])
        unique = tuple(range(54))
        for action in cube3.ACTIONS:
            state = unique
            for _ in range(4):
                state = cube3.step(state,action)
            self.assertEqual(state,unique)
            turned = cube3.step(unique,action)
            self.assertEqual([turned[i*9+4] for i in range(6)],[i*9+4 for i in range(6)])
            self.assertEqual(sorted(turned),list(unique))

    def test_harder_cases_preserve_contract_and_replay_witnesses(self):
        config = json.loads((ROOT/'experiments/task_text_baselines/configs/difficulty_ladder.json').read_text())
        config["cases_per_level"] = 2
        for task in config["tasks"]:
            for case in prepare_level(task,0,config["tasks"][task],config):
                verdict = judge(case,json.dumps({"actions":case["reference"]["actions"],"final_status":"solved"}))
                self.assertTrue(verdict["contract_pass"],(case["id"],verdict))
                self.assertNotIn("board_after",case["prompt"])
                if task == 'puzzle15':
                    self.assertIn("4x4",case["prompt"])
                    self.assertIn("1 through 15",case["prompt"])
                    self.assertNotIn("Action examples",case["prompt"])
                    self.assertLessEqual(puzzle15.manhattan(case["initial"]),case["reference"]["upper_bound"])
                if task == 'cube3':
                    self.assertIn("3x3",case["prompt"])
                    self.assertIn("[After B']",case["prompt"])
                if task == 'blocks':
                    self.assertIn("6=011/110",case["prompt"])


if __name__ == '__main__':
    unittest.main()
