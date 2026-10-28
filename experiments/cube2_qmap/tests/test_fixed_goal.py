"""Fixed-goal supervision must not leak heldout labels via candidates."""
import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from experiments.cube2_qmap.src.fixed_goal import FixedGoalReplicas, tensor_batch, balanced_parents, exhaustive_policy, failure_pool, make_batch, path_relations, prepare, reach_within_budget, validate_resume_contract
from experiments.cube2_qmap.src.full_state_joint import batch_losses
from experiments.puzzle_qmap.src.model import QMap
from experiments.cube2_qmap.tests.test_full_state_joint import Ring


class FixedGoalTests(unittest.TestCase):
    def setUp(self):
        self.env = Ring()
        self.env.depth = self.env.distance(np.arange(self.env.size), 0).astype(np.uint8)

    def test_failure_replay_uses_wrong_complete_training_stars_and_basin_sizes(self):
        env=self.env;ids=np.arange(env.size);children=env.successors(ids)
        pointer=children[ids,env.depth[children].argmin(-1)];pointer[0]=0
        pointer[4],pointer[10]=5,11
        parents,weights,report=failure_pool(env,np.zeros(env.size,dtype=bool),pointer)
        np.testing.assert_array_equal(parents,[4,10])
        from experiments.cube2_qmap.src.policy_audit import two_cycle_basins
        labels=two_cycle_basins(pointer)
        expected=1+np.log1p([(labels==p).sum() for p in parents]);expected/=expected.sum()
        np.testing.assert_allclose(weights,expected)
        self.assertEqual(report['cycle_exits'],2)
        heldout=np.zeros(env.size,dtype=bool);heldout[5]=True
        parents,weights,_=failure_pool(env,heldout,pointer)
        np.testing.assert_array_equal(parents,[10]);np.testing.assert_array_equal(weights,[1])
        pointer[4],pointer[10]=3,9
        parents,weights,_=failure_pool(env,heldout,pointer)
        self.assertEqual(len(parents),0);self.assertEqual(len(weights),0)

    def test_device_stars_and_replica_update_preserve_original_objective(self):
        torch.set_num_threads(1);torch.manual_seed(19)
        parents=np.arange(self.env.size);heldout=np.zeros(self.env.size,dtype=bool);heldout[[5,10]]=True
        original=make_batch(self.env,parents,heldout)
        packed=tensor_batch(torch.as_tensor(parents),torch.as_tensor(self.env.successors(parents)),
                            torch.as_tensor(self.env.depth),torch.as_tensor(heldout))
        for key in ('states','pairs','labels','children','goals','truth'):
            np.testing.assert_array_equal(packed[key],original[key])
        model=QMap(self.env.size,dimension=8,hidden=16,metric='linf',cap=self.env.diameter).double()
        replica=deepcopy(model);a=torch.optim.Adam(model.parameters(),lr=.0003);b=torch.optim.Adam(replica.parameters(),lr=.0003)
        config=dict(training=dict(temperature=1.,rank_weight=1.,clip_norm=5.))
        parallel=FixedGoalReplicas(replica,self.env,heldout,['cpu','cpu'])
        try:
            for _ in range(3):
                h,r=batch_losses(model,torch.as_tensor(self.env.features(original['states'])).double(),original,config)
                a.zero_grad();(h+r).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);a.step()
                ph,pr,_=parallel.update(parents,b,config)
                torch.testing.assert_close(ph+pr,(h+r).detach(),rtol=1e-6,atol=1e-6,check_dtype=False)
                for x,y in zip(model.parameters(),replica.parameters()):
                    torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=1e-7)
                    torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-7)
        finally:parallel.close()

    def test_cycle_focused_replay_preserves_correct_relations_and_total_quota(self):
        env=self.env;ids=np.arange(env.size);children=env.successors(ids)
        pointer=children[ids,env.depth[children].argmin(-1)];pointer[0]=0
        pointer[[4,7,8,9,10,20]]=[5,8,9,10,11,21]
        parents,weights,report=failure_pool(env,np.zeros(env.size,bool),pointer,
            predicted=env.depth.astype(float)*.1,retention_fraction=.25,cycle_fraction=2/3)
        from experiments.cube2_qmap.src.policy_audit import two_cycle_basins
        labels=two_cycle_basins(pointer)
        wrong=env.depth[pointer[parents]].astype(int)!=env.depth[parents].astype(int)-1
        cycle=wrong&(pointer[pointer[parents]]==parents)
        self.assertAlmostEqual(float(weights[cycle].sum()),.5)
        self.assertAlmostEqual(float(weights[wrong&~cycle&(labels[parents]>=0)].sum()),.25)
        self.assertAlmostEqual(float(weights[~wrong].sum()),.25)
        self.assertEqual(float(weights[wrong&(labels[parents]==-1)].sum()),0.)
        self.assertAlmostEqual(report['cycle_probability'],.5)
        before={'training':{'epochs':500,'learning_rate':.0001}}
        after=deepcopy(before);after['training']['failure_cycle_fraction']=2/3
        validate_resume_contract(before,after,True)
        with self.assertRaises(ValueError):validate_resume_contract(before,after)
        for key,value in [('epochs',501),('learning_rate',.0003)]:
            changed=deepcopy(after);changed['training'][key]=value
            with self.assertRaises(ValueError):validate_resume_contract(before,changed,True)

    def test_fragile_correct_replay_keeps_original_total_quota_and_training_boundary(self):
        env=self.env;ids=np.arange(env.size);height=env.depth.astype(float)
        height[3]=10.;height[6]=4.15
        children=env.successors(ids);pointer=children[ids,height[children].argmin(-1)];pointer[0]=0
        held=np.zeros(env.size,dtype=bool)
        parents,weights,report=failure_pool(env,held,pointer,predicted=height,retention_fraction=.25)
        correct=env.depth[pointer[parents]].astype(int)==env.depth[parents].astype(int)-1
        self.assertTrue(correct.any());self.assertTrue((~correct).any())
        self.assertAlmostEqual(float(weights[correct].sum()),.25)
        self.assertAlmostEqual(float(weights[~correct].sum()),.75)
        self.assertEqual(int(correct.sum()),report['retained_correct_stars'])
        self.assertEqual(len(parents),len(np.unique(parents)))
        self.assertFalse(held[parents].any());self.assertFalse(held[env.successors(parents)].any())
        held[5]=True
        parents,weights,report=failure_pool(env,held,pointer,predicted=height,retention_fraction=.25)
        self.assertFalse(held[parents].any());self.assertFalse(held[env.successors(parents)].any())
        if len(weights):self.assertAlmostEqual(float(weights.sum()),1.)
        else:self.assertEqual(report['wrong_training_stars'],0)

    def test_heldout_labels_are_absent_from_both_loss_families(self):
        heldout = np.zeros(self.env.size, dtype=bool)
        heldout[[5, 10]] = True
        parents = np.arange(self.env.size)
        batch = make_batch(self.env, parents, heldout)
        pairs = batch["states"][batch["pairs"]]
        self.assertFalse(heldout[pairs[:, 0]].any())
        np.testing.assert_array_equal(pairs[:, 1], np.zeros(len(pairs)))
        np.testing.assert_array_equal(batch["labels"], self.env.depth[pairs[:, 0]])
        children = batch["states"][batch["children"]]
        expected = self.env.successors(parents[(~heldout[np.column_stack((parents, self.env.successors(parents)))]).all(-1) & (parents != 0)])
        np.testing.assert_array_equal(children, expected)
        np.testing.assert_array_equal(batch["truth"], self.env.depth[children])
        self.assertFalse(heldout[children].any())

    def test_split_and_balanced_replay_keep_every_layer_and_goal_trainable(self):
        env = SimpleNamespace(size=201, diameter=5, depth=np.r_[0, np.repeat(np.arange(1, 6), 40)])
        config = {"data": {"seed": 7, "first_heldout_depth": 4, "heldout_fraction_per_split": .1,
                           "minimum_heldout_per_layer": 0}, "evaluation": {"states_per_layer": 2}}
        mask, layers, cohorts = prepare(env, config)
        self.assertEqual(mask[0], 0)
        self.assertTrue(all(len(layer) for layer in layers))
        self.assertEqual(int((mask == 1).sum()), 8)
        self.assertEqual(int((mask == 2).sum()), 8)
        self.assertTrue((mask[cohorts["validation"]] == 1).all())
        self.assertTrue((mask[cohorts["test"]] == 2).all())
        replay = balanced_parents(layers, 60, 0, np.random.default_rng(5))
        self.assertFalse(mask[replay].any())
        np.testing.assert_array_equal(np.bincount(env.depth[replay])[1:], np.full(5, 12))

    def test_policy_scoring_counts_cycles_and_budget_limits_without_correcting_actions(self):
        policy = np.array([0, 0, 1, 2, 5, 4, 5])
        np.testing.assert_array_equal(reach_within_budget(policy, 0, 2), [0, 1, 2, -1, -1, -1, -1])
        np.testing.assert_array_equal(reach_within_budget(policy, 0, 3), [0, 1, 2, 3, -1, -1, -1])

    def test_intermediate_nonadjacent_pairs_are_exact_and_heldout_endpoints_excluded(self):
        heldout = np.zeros(self.env.size, dtype=bool)
        heldout[3] = True
        pairs, labels = path_relations(self.env, np.array([5]), heldout, np.random.default_rng(7))
        self.assertTrue(np.any((pairs[:, 0] == 5) & (pairs[:, 1] == 2)))
        self.assertFalse(heldout[pairs].any())
        np.testing.assert_array_equal(labels, self.env.distance(*pairs.T))
        batch = make_batch(self.env, np.arange(self.env.size), heldout, np.array([5, 10]), np.random.default_rng(8))
        physical = batch["states"][batch["pairs"]]
        self.assertGreater(batch["path_pair_count"], 0)
        self.assertFalse(heldout[physical].any())
        np.testing.assert_array_equal(batch["labels"], self.env.distance(*physical.T))
        model = QMap(self.env.size, dimension=8, hidden=16, metric="linf", cap=self.env.diameter)
        h, r = batch_losses(model, torch.as_tensor(self.env.features(batch["states"])), batch,
                            {"training": {"temperature": 1.0}})
        (h + r).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_exhaustive_policy_chooses_from_network_without_oracle_correction(self):
        class ConstantQ:
            def encode(self, features):
                return torch.zeros((len(features), 2))

            def distance(self, a, b):
                return (a - b).abs().amax(-1)
        # Every action ties, so choose the first (+) and follow it unchanged.
        result = exhaustive_policy(ConstantQ(), self.env, np.zeros(self.env.size), {"evaluation": {"horizon": 3}}, "cpu")
        self.assertEqual(result["cohorts"]["all_states"]["solved"], 4)


if __name__ == "__main__":
    unittest.main()
