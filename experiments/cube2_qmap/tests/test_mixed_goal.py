import unittest
import copy

import numpy as np

from experiments.cube2_qmap.src.mixed_goal import HeldoutStates, global_cohorts, restore_continuation
from experiments.cube2_qmap.src.full_state_joint import FullStateSampler
from experiments.cube2_qmap.tests.test_full_state_joint import Ring


class MixedGoalTests(unittest.TestCase):
    def setUp(self):
        self.env = Ring()
        self.env.depth = self.env.distance(np.arange(self.env.size), 0)
        self.layers = [np.flatnonzero(self.env.depth == d) for d in range(self.env.diameter + 1)]
        self.mask = np.zeros(self.env.size, dtype=np.uint8)
        self.mask[[5, 15]] = 1
        self.mask[[10, 30]] = 2
        self.config = {"data": {"seed": 3, "target_distance_levels": list(range(1,21))},
                       "training": {"parent_batch_size": 8},
                       "evaluation": {"global_distance_levels": list(range(1,21)),
                                      "global_validation_per_level": 3, "global_test_per_level": 4}}

    def test_global_supervision_respects_restoration_mask_for_edges_goals_and_all_candidates(self):
        heldout = self.mask > 0
        guard = HeldoutStates(heldout)
        sampler = FullStateSampler(self.env, self.config, guard, self.layers)
        parents = np.arange(self.env.size)
        stars = np.column_stack((parents, self.env.successors(parents)))
        eligible = parents[~heldout[stars].any(-1)]
        batch = sampler.draw(eligible)
        physical = batch["states"][batch["pairs"]]
        self.assertFalse(heldout[physical].any())
        np.testing.assert_array_equal(batch["labels"], self.env.distance(*physical.T))
        candidates = batch["states"][batch["children"]]
        goals = batch["states"][batch["goals"]]
        self.assertFalse(heldout[candidates].any())
        self.assertFalse(heldout[goals].any())
        np.testing.assert_array_equal(batch["truth"], self.env.distance(candidates, goals[:,None]))

    def test_global_evaluation_targets_are_protected_from_all_pretraining_fixed_goal_labels(self):
        cohorts = global_cohorts(self.env, self.mask, self.config, self.layers)
        for name, code in (("validation", 1), ("test", 2)):
            tasks = cohorts[name]
            self.assertTrue((self.mask[tasks[:,1]] == code).all())
            distances = self.env.distance(*tasks.T)
            count = self.config["evaluation"][f"global_{name}_per_level"]
            np.testing.assert_array_equal(distances,np.repeat(np.arange(1,21),count))

    def test_legacy_epoch_resume_restores_next_permutation_targets_and_counters(self):
        config = copy.deepcopy(self.config)
        config.update(model={"dimension": 8}, seed=5, base_contract="fixed_goal.json")
        config["training"].update(epochs=5, learning_rate=.0003)
        guard = HeldoutStates(np.zeros(self.env.size, dtype=bool))
        original = FullStateSampler(self.env, config, guard, self.layers)
        for parents in original.batches():
            stars = np.column_stack((parents,self.env.successors(parents)))
            eligible = parents[~guard.heldout[stars].any(-1)]
            if len(eligible): original.draw(eligible)
        rng = np.random.default_rng(41)
        rng.integers(100,size=15)
        saved = {"config":config,"epoch":1,"step":6,"smoke_steps":0,
                 "rng":copy.deepcopy(rng.bit_generator.state),
                 "global_rng":copy.deepcopy(original.rng.bit_generator.state),
                 "global_seen":original.seen.copy()}
        row = {"epoch":1,"step":6,"global_coverage":original.coverage()}
        prior = {"evaluations":[row]}
        extended = copy.deepcopy(config);extended["training"]["epochs"] = 100
        restored = FullStateSampler(self.env, extended, guard, self.layers)
        restored_rng = np.random.default_rng(99)
        restore_continuation(saved,prior,extended,self.env,restored,restored_rng)
        self.assertEqual(restored.coverage(),original.coverage())
        np.testing.assert_array_equal(rng.integers(100,size=20),restored_rng.integers(100,size=20))
        first_a, first_b = next(original.batches()), next(restored.batches())
        np.testing.assert_array_equal(first_a,first_b)
        stars = np.column_stack((first_a,self.env.successors(first_a)))
        eligible = first_a[~guard.heldout[stars].any(-1)]
        a,b = original.draw(eligible),restored.draw(eligible)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        changed = copy.deepcopy(extended);changed["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError,"epoch budget"):
            restore_continuation(saved,prior,changed,self.env,restored,restored_rng)
        saved["step"] = 5
        with self.assertRaisesRegex(ValueError,"completed formal epoch"):
            restore_continuation(saved,prior,extended,self.env,restored,restored_rng)


if __name__ == "__main__":
    unittest.main()
