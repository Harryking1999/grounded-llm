"""Interventions preserve addresses/history and change only executed early control."""

from types import SimpleNamespace
import unittest

import torch

from experiments.flamingo_map_reader.src.counterfactual import reverse_candidate_q
from experiments.flamingo_map_reader.src.diagnostic_eval import AssistedSession, reversed_reference, safe_choice
from experiments.flamingo_map_reader.src.diagnostic_queue import owned_jobs
from experiments.flamingo_map_reader.src.graph import GraphStep
from experiments.flamingo_map_reader.src.memory import MapBatch
from experiments.flamingo_map_reader.src.trajectory_eval import closed_loop


def step(current=0, goal=2, distances=(1., 3., 2., 2.)):
    count = len(distances)
    vectors = torch.tensor([[[10., 0.], [20., 0.], *[[float(i), float(d)] for i, d in enumerate(distances)]]])
    batch = MapBatch(vectors, torch.tensor([[0, 1, *([2] * count)]]),
                     torch.tensor([[0, 0, *range(1, count + 1)]]), torch.ones((1, count + 2), dtype=torch.bool))
    best = tuple(i for i, d in enumerate(distances, 1) if d == min(distances)) if count else ()
    return GraphStep(current, goal, batch, tuple(range(10, 10 + count)),
                     tuple(range(current + 1, current + count + 1)), 2., tuple(distances), best, current == goal)


class TinyEnvironment:
    config = dict(data={})

    def remaining(self, current, goal, path=None):
        return goal - current

    def chosen_is_shortest(self, step, chosen, path, remaining):
        return chosen is not None and 1 <= chosen <= len(step.candidate_actions) and step.candidate_destinations[chosen - 1] == step.current + 1

    def chosen_keeps_reachable(self, step, chosen, path, remaining):
        return self.chosen_is_shortest(step, chosen, path, remaining)

    def reachable_candidates(self, step, path, remaining):
        return 1

    def step(self, path, goal, rng):
        return step(path[-1], goal, () if path[-1] == goal else (1.,))

    def update(self, step, path, actions):
        return 'actual path: ' + str(path)

    def terminal(self, path, actions):
        return '<done/>'

    def execute(self, step, chosen):
        return step.candidate_actions[chosen - 1], step.candidate_destinations[chosen - 1]


class FakeSession:
    def __init__(self, answers):
        self.answers, self.accepted, self.steps, self.seen = iter(answers), [], [], []
        self.prefix, self.ids = [1, 2], [0, 0]

    def ask(self, user, step):
        self.steps.append(step)
        self.seen.append([s.map_batch.vectors.clone() for s in self.steps])
        return next(self.answers), dict(generated_tokens=1, seconds=0., context_exhausted=False)

    def accept(self, answer):
        self.accepted.append(answer)


class FinalDiagnosticsTest(unittest.TestCase):
    def test_reverse_is_distance_ranked_not_slot_reversal(self):
        original = step()
        altered, sources = reverse_candidate_q(original)
        self.assertEqual(sources, (1, 0, 3, 2))
        self.assertEqual(altered.candidate_map_distances, (3., 1., 2., 2.))
        self.assertEqual(altered.map_minimal_candidates, (2,))
        self.assertEqual(altered.candidate_actions, original.candidate_actions)
        self.assertEqual(altered.candidate_destinations, original.candidate_destinations)
        torch.testing.assert_close(altered.map_batch.vectors[:, :2], original.map_batch.vectors[:, :2])
        for name in ('roles', 'candidate_ids', 'valid'):
            torch.testing.assert_close(getattr(altered.map_batch, name), getattr(original.map_batch, name))
        torch.testing.assert_close(original.map_batch.vectors[:, 2, :], torch.tensor([[0., 1.]]))

    def test_reverse_is_involution_including_ties_and_terminal(self):
        for distances in ((1., 3., 2., 2.), (1.,), (), (1., 1., 1.)):
            original = step(distances=distances)
            altered, _ = reverse_candidate_q(original)
            restored, _ = reverse_candidate_q(altered)
            # Tied contents need not return to their original slots under a
            # distance/ID tie break; the represented distances always do.
            self.assertEqual(restored.candidate_map_distances, original.candidate_map_distances)
            if len(set(distances)) == len(distances):
                torch.testing.assert_close(restored.map_batch.vectors, original.map_batch.vectors)

    def test_reference_restores_original_historical_maps_and_gold_text(self):
        original = step(distances=(1., 3.))
        demo = SimpleNamespace(turns=[SimpleNamespace(step=original, executed_path=(0,), user_text='same', answer_text='gold')]*2)
        reverse = FakeSession(['<action>2</action>'] * 2)
        baseline = dict(variant=0, turns=[dict(answer='<action>1</action>', current='0', goal='2', remaining_shortest=2, candidates=2)]*2)
        result = reversed_reference(reverse, demo, TinyEnvironment(), baseline)
        self.assertEqual(reverse.accepted, ['gold', 'gold'])
        torch.testing.assert_close(reverse.seen[1][0], original.map_batch.vectors)
        self.assertFalse(torch.equal(reverse.seen[1][1], original.map_batch.vectors))
        self.assertTrue(result['turns'][0]['normal']['action_keeps_goal_reachable'])
        self.assertFalse(result['turns'][0]['reverse']['action_keeps_goal_reachable'])
        self.assertTrue(result['turns'][0]['reverse_presented_map_minimum'])

    def test_helper_executes_safe_action_but_history_retains_model_answer(self):
        model = FakeSession(['<action>999</action>', '<action>1</action>', '<done/>'])
        session = AssistedSession(model, TinyEnvironment(), 1)
        value = closed_loop(session, dict(start=0, goal=2, sample_seed=1, shortest_moves=2), '',
                            TinyEnvironment(), dict(maximum_demonstration_actions=5, data={}), 0)
        self.assertTrue(value['reached_goal'])
        self.assertEqual(value['actual_path'], ['0', '1', '2'])
        self.assertEqual(model.accepted[0], '<action>999</action>')
        self.assertEqual(session.answers[0]['executed_candidate_id'], 1)
        self.assertEqual([r['assisted'] for r in session.answers], [True, False, False])

    def test_safe_policy_is_invariant_to_candidate_numbering(self):
        original = step(distances=(1., 3.))
        from dataclasses import replace
        both = TinyEnvironment()
        both.chosen_is_shortest = lambda *args: True
        original = replace(original, candidate_actions=(20, 10))
        self.assertEqual(safe_choice(both, original, [0]), 2)

    def test_workers_are_disjoint_and_cover_both_stages(self):
        stage = list(range(11))
        for index in (0, 1):
            left, right = owned_jobs(stage, index, 0), owned_jobs(stage, index, 1)
            self.assertFalse(set(left) & set(right))
            self.assertEqual(sorted(left + right), stage)


if __name__ == '__main__':
    unittest.main()
