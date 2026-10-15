import json
from pathlib import Path
import unittest
import torch

from experiments.gcml_counterexamples.src import blocks as rules
from experiments.map_sft_baselines.src.common import demonstration, renumber_trajectory, sorted_step
from experiments.map_sft_baselines.src.train import learning_rate_factor, validate_resume_contract


class EpochRenumberTest(unittest.TestCase):
    def test_ids_change_but_physical_supervision_and_memory_stay_aligned(self):
        class Map:
            metric = 'euclidean'
            def encode(self, states):
                return torch.tensor([[float((s >> i) & 1) for i in range(100)] for s in states])
        state = (1 << 100)-1
        first = sorted_step(Map(), state, 0)
        action, goal = first.execute(17)
        steps = [sorted_step(Map(), state, goal), sorted_step(Map(), goal, goal)]
        record = dict(start=str(state), goal=str(goal), route_ids=[17])
        original = demonstration(record, None, True, 'ranking_first', steps)
        a = renumber_trajectory(record, steps, 13, 0, 7)
        repeated = renumber_trajectory(record, steps, 13, 0, 7)
        b = renumber_trajectory(record, steps, 13, 1, 7)
        self.assertEqual(a.turns[0].step.candidate_actions, repeated.turns[0].step.candidate_actions)
        self.assertNotEqual(a.turns[0].step.candidate_actions, b.turns[0].step.candidate_actions)
        for demo in (a, b):
            self.assertEqual(demo.executed_path, original.executed_path)
            turn = demo.turns[0]
            self.assertEqual(turn.step.execute(turn.chosen_id), (action, goal))
            self.assertEqual(turn.answer_text.splitlines()[1], original.turns[0].answer_text.splitlines()[1])
            self.assertEqual(turn.answer_text.splitlines()[-1], f'<action>{turn.chosen_id}</action>')
            for local, physical in enumerate(turn.step.candidate_actions, 1):
                old = steps[0].candidate_actions.index(physical)
                self.assertTrue(torch.equal(turn.step.map_batch.vectors[0,local+1], steps[0].map_batch.vectors[0,old+2]))
                self.assertEqual(turn.step.map_batch.candidate_ids[0,local+1].item(), local)
                self.assertEqual(turn.step.candidate_map_distances[local-1], steps[0].candidate_map_distances[old])
                move = rules.PLACEMENTS[physical][1]
                self.assertIn(f"{local}: remove({move['shape_id']},{move['row']},{move['col']})", turn.user_text)
            best = turn.step.map_minimal_candidates[0]
            self.assertTrue(turn.answer_text.startswith(f'Map-distance ranking to the goal, closest to farthest: {best}'))
            self.assertEqual(demo.turns[-1].answer_text, original.turns[-1].answer_text)

    def test_constant_schedule_and_resume_contract(self):
        cfg = json.loads((Path(__file__).parents[1]/'configs/study.json').read_text())
        t = dict(cfg['training'], **cfg['baselines']['B1_const_renumber']['training'])
        for step, expected in [(0,0.), (3125,.5), (6250,1.), (125000,1.), (250000,1.)]:
            self.assertEqual(learning_rate_factor(step, 250000, t), expected)
        self.assertAlmostEqual(learning_rate_factor(250000,250000,cfg['training']),.1)
        self.assertEqual(learning_rate_factor(6250,250000,cfg['training']),.5)
        saved = dict(config=json.loads(json.dumps(cfg)), baseline='B1_const_renumber')
        cfg['baselines']['B1_const_renumber']['renumber_each_epoch'] = False
        with self.assertRaises(ValueError):validate_resume_contract(saved,cfg,'B1_const_renumber')


if __name__ == '__main__':unittest.main()
