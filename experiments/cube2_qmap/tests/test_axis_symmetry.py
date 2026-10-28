import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.cube2_qmap.src.environment import Cube2
from grounded_llm.tasks.cube2.model import MODEL_TYPE, MapReader, QMap


class AxisSymmetryTests(unittest.TestCase):
    def test_inverse_feature_projection_and_encoder_commute_with_axis_actions(self):
        from grounded_llm.tasks.cube2.symmetry import inverse_feature_indices
        env=Cube2(metric='QTM_URF_6');ids=np.random.default_rng(54).integers(env.size,size=512)
        inv=env.relative(ids,0);features=env.features(ids);gather=inverse_feature_indices()
        np.testing.assert_array_equal(features[...,gather],env.features(inv))
        np.testing.assert_array_equal(gather[gather],np.arange(192))
        torch.set_num_threads(1)
        model=QMap(192,128,16,14,'linf',symmetry='cube_axis_permutation_average',inverse_symmetry=True)
        bits=torch.as_tensor(features);inverse=torch.as_tensor(env.features(inv))
        q=model.encode(bits);qi=model.encode(inverse)
        self.assertEqual(q.shape,(512,128));torch.testing.assert_close(q,qi,rtol=0,atol=0)
        for axis in range(6):
            transformed=torch.as_tensor(env.features(env.axis_symmetry(ids,axis)))
            torch.testing.assert_close(model.encode(transformed),q[...,model.axis_output_action[axis]],rtol=1e-5,atol=1e-6)
        model.encode(bits[:4]).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'projected.pt'
            torch.save(dict(model_type=MODEL_TYPE,task=env.name,actions=list(env.actions),cap=14,
                model_config=dict(dimension=128,hidden=16,metric='linf',symmetry='cube_axis_permutation_average',inverse_symmetry=True),
                model=model.state_dict()),checkpoint)
            reader=MapReader(env,checkpoint,coordinate_frame='goal_relative')
            torch.testing.assert_close(reader.encode(ids),q,rtol=0,atol=0)
            self.assertFalse(reader.inverse_average)
            children=env.successors(ids[:8]);scores=reader.scores(children.ravel(),[0]).reshape(8,6)
            self.assertTrue(np.isfinite(scores).all())
            with self.assertRaises(ValueError):MapReader(env,checkpoint)

    def test_goal_relative_reader_preserves_actions_and_restoration_without_oracle(self):
        env=Cube2(metric="QTM_URF_6")
        source,goal=np.random.default_rng(32).integers(env.size,size=(2,1000))
        relative=env.relative(goal,source)
        np.testing.assert_array_equal(env.relative(goal[:,None],env.successors(source)),env.successors(relative))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'q.pt';model=QMap(env.feature_dim,8,16,14,'linf')
            torch.save(dict(model_type=MODEL_TYPE,task=env.name,actions=list(env.actions),
                model_config=dict(dimension=8,hidden=16,metric='linf'),cap=14,model=model.state_dict()),checkpoint)
            absolute=MapReader(env,checkpoint)
            adapted=MapReader(env,checkpoint,coordinate_frame='goal_relative')
            np.testing.assert_array_equal(adapted.scores(source,goal),absolute.scores(relative,[0]))
            np.testing.assert_array_equal(adapted.scores(source,[0]),absolute.scores(source,[0]))
            for s,g,r in zip(source[:10],goal[:10],relative[:10]):
                action,child=adapted.choose(int(s),int(g))
                self.assertEqual(action,absolute.choose(int(r),0)[0])
                self.assertEqual(child,env.step(int(s),action))

    def test_inverse_average_is_symmetric_and_keeps_encodings_and_all_successors(self):
        env=Cube2(metric='QTM_URF_6')
        source,goal=np.random.default_rng(41).integers(env.size,size=(2,64))
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'q.pt';model=QMap(env.feature_dim,128,16,14,'linf')
            torch.save(dict(model_type=MODEL_TYPE,task=env.name,actions=list(env.actions),
                model_config=dict(dimension=128,hidden=16,metric='linf'),cap=14,model=model.state_dict()),checkpoint)
            ordinary=MapReader(env,checkpoint,coordinate_frame='goal_relative')
            averaged=MapReader(env,checkpoint,coordinate_frame='goal_relative',inverse_average=True)
            np.testing.assert_array_equal(averaged.scores(source,goal),averaged.scores(goal,source))
            torch.testing.assert_close(averaged.encode(source),ordinary.encode(source),rtol=0,atol=0)
            children=env.successors(source);targets=np.broadcast_to(goal[:,None],children.shape)
            scores=averaged.scores(children.ravel(),targets.ravel()).reshape(children.shape)
            self.assertEqual(scores.shape,(64,6));self.assertTrue(np.isfinite(scores).all())
            for i,(s,g) in enumerate(zip(source[:5],goal[:5])):
                action,child=averaged.choose(int(s),int(g))
                self.assertEqual(child,env.step(int(s),action))
                self.assertEqual(action,env.actions[int(scores[i].argmin())])
            with self.assertRaises(ValueError):MapReader(env,checkpoint,inverse_average=True)

    def test_all_six_maps_preserve_solved_and_relabel_every_action_transition(self):
        for metric in ("QTM_URF_6", "HTM_URF_9"):
            env = Cube2(metric=metric)
            rng = np.random.default_rng(30)
            states = rng.integers(env.size, size=3000)
            for transform in range(6):
                self.assertEqual(env.axis_symmetry(0, transform), 0)
                changed = env.axis_symmetry(states, transform)
                self.assertTrue(np.all((changed >= 0) & (changed < env.size)))
                expected = env.successors(changed)[:, env.axis_action_map[transform]]
                actual = env.axis_symmetry(env.successors(states), transform)
                np.testing.assert_array_equal(actual, expected)
                inverse = next(t for t in range(6) if np.array_equal(
                    env.axis_symmetry(changed[:20], t), states[:20]))
                np.testing.assert_array_equal(env.axis_symmetry(changed, inverse), states)

    def test_relative_relations_and_broadcasted_packs_transform_together(self):
        env = Cube2(metric="QTM_URF_6")
        rng = np.random.default_rng(31)
        source, goal = rng.integers(env.size, size=(2, 1000))
        changed = env.axis_symmetry(np.stack((source, goal), -1)[:, None, :],
                                    np.arange(6)[None, :, None])
        expected = env.axis_symmetry(env.relative(source, goal)[:, None], np.arange(6))
        actual = env.relative(changed[..., 0], changed[..., 1])
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
