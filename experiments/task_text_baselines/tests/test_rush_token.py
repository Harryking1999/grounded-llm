"""Check source conventions, optimal labels, and physical-plan failure modes."""
from collections import deque
import itertools
import json
import unittest
from grounded_llm.tasks.rush_hour.environment import RushHour
from grounded_llm.tasks.token_swapping.environment import TokenSwapping
from grounded_llm.tasks.text import judge, prompt, section_text
from experiments.task_text_baselines.src.prepare_rush_token import decode_public_tokens


class RulesTests(unittest.TestCase):
    def test_rush_cannot_jump_or_rotate(self):
        env = RushHour('............AA.BB...................')
        with self.assertRaises(ValueError):
            env.apply(env.start, {'vehicle':'A','displacement':4})
        for action in ({'vehicle':'A','displacement':True}, {'vehicle':'A','displacement':0},
                       {'vehicle':'missing','displacement':1}):
            with self.assertRaises(ValueError):
                env.apply(env.start, action)

    def test_rush_public_distance_and_component_size(self):
        env = RushHour('GBBoLoGHIoLMGHIAAMCCCKoMooJKDDEEJFFo')
        actions, size = env.solve()
        self.assertEqual((len(actions), size), (51, 4780))
        state = env.start
        for action in actions:
            previous = state
            state, cost = env.apply(state, action)
            reverse = dict(action,displacement=-action['displacement'])
            self.assertEqual(env.apply(state, reverse), (previous,1))
            self.assertEqual(cost,1)
        self.assertTrue(env.terminal(state))

    def test_bidirectional_cost_matches_independent_complete_bfs(self):
        for edges in ([[0,1],[1,2],[2,3],[3,4]], [[0,1],[0,2],[0,3],[0,4]],
                      [[0,1],[1,2],[2,3],[3,4],[0,4]]):
            goal = tuple(range(5))
            distances, queue = {goal:0}, deque([goal])
            while queue:
                state=queue.popleft()
                for a,b in edges:
                    child=list(state)
                    child[a],child[b]=child[b],child[a]
                    child=tuple(child)
                    if child not in distances:
                        distances[child]=distances[state]+1
                        queue.append(child)
            env=TokenSwapping(5,edges)
            for state in itertools.permutations(range(5)):
                actions,_=env.solve(state)
                self.assertEqual(len(actions),distances[state])
                for action in actions:
                    state,_=env.apply(state,action)
                self.assertEqual(state,goal)

    def test_upstream_inverse_permutation_and_partial_filter(self):
        source='solutions["example"] = {"1:2", "0112", "0112_01"};'
        cases=list(decode_public_tokens(source))
        self.assertEqual(len(cases),1)
        _,_,edges,initial,actions=cases[0]
        self.assertEqual(initial,[2,0,1])
        env=TokenSwapping(3,edges)
        state=tuple(initial)
        for action in actions:
            state,_=env.apply(state,action)
        self.assertEqual(state,env.goal)

    def test_judge_separates_completion_from_optimality_and_illegality(self):
        case=dict(task='token_swapping',edges=[[0,1],[1,2]],initial=[1,0,2],
                  action_limit=256,reference={'cost':1})
        output={'actions':[[0,1]]*3,'final_status':'solved'}
        verdict=judge(case,json.dumps(output))
        self.assertTrue(verdict['contract_pass'])
        self.assertFalse(verdict['optimal'])
        self.assertEqual(verdict['excess_cost'],2)
        output['actions']=[[0,2]]
        self.assertEqual(judge(case,json.dumps(output))['failure'],'illegal_action')
        output['actions']=[[False,1]]
        self.assertEqual(judge(case,json.dumps(output))['failure'],'illegal_action')

    def test_new_text_contracts_render(self):
        examples=[dict(task='token_swapping',edges=[[0,1]],initial=[1,0],action_limit=256),
                  dict(task='rush_hour',initial='............AA......................',action_limit=256)]
        for case in examples:
            self.assertNotIn('{action_limit}',prompt(case))
            for heading in ('共享任务描述与规则','地图组／CoT 组','SOTA 组／32B 组','监督答案'):
                self.assertTrue(section_text(case['task'],heading))


if __name__=='__main__':
    unittest.main()
