"""Numerical deduplication must preserve complete arithmetic continuations."""
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.physical_data import save_scene
from grounded_llm.tasks.game24.environment import Game24
from grounded_llm.tasks.game24.relation_data import GameScene,GameCorpus
from experiments.new_tasks_qmap.src.game_finite_evaluate import numerical_queries


class FiniteGameTests(unittest.TestCase):
    def test_deduplication_keeps_exact_reachability_and_both_targets(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            env=Game24(4);scenes=[]
            for i,numbers in enumerate([(1,3,4,6),(1,2,3,4)]):
                graph=env.graph(numbers)
                values=[tuple(Fraction(v) for v in row) for row in graph['metadata']['exact_rational_states']]
                children=graph['children']
                for target in (21,24):
                    if (target,) not in values:
                        values.append((Fraction(target),));children=np.vstack([children,np.full((1,len(env.actions)),-1,dtype=np.int32)])
                goals=[values.index((t,)) for t in (21,24)];path=Path(directory)/str(i)
                record=save_scene(path,np.arange(len(values))[:,None],children,goals,
                    dict(task='game24',split='train',max_numbers=4,training_goals=2,original_state=0,source=dict(numbers=numbers)),state_dtype=np.int32)
                atomic_json(path/'values.json',[[str(v) for v in row] for row in values])
                scene=GameScene(path,record);scenes.append(scene)
                for name in ('states','children','distances','reverse_distances'):stack.callback(getattr(scene,name)._mmap.close)
            corpus=GameCorpus(scenes);exposure=np.ones(corpus.size,dtype=np.int64)
            groups,coverage=numerical_queries(corpus,[21,24],exposure)
            self.assertEqual(coverage['numerical_input_coverage'],1)
            self.assertLess(coverage['exact_numerical_states'],corpus.size)
            for target in (21,24):
                self.assertEqual(len(groups[f'root4_to_{target}']),2)
                for arity in (2,3):
                    rows=groups[f'suffix{arity}_to_{target}'];seen=[]
                    for source,goal in rows:
                        sid=int(corpus.locate(source));scene=scenes[sid];offset=corpus.offsets[sid]
                        value=scene.values[source-offset];seen.append(value)
                        self.assertEqual(scene.values[goal-offset],(target,))
                    expected={v for scene in scenes for v in scene.values if len(v)==arity}
                    self.assertEqual(len(seen),len(set(seen)));self.assertEqual(set(seen),expected)
                    labels=corpus.distance(*rows.T);finite=np.isfinite(labels)
                    self.assertTrue(finite.any());self.assertTrue((~finite).any())
                    self.assertTrue((labels[finite]==arity-1).all())
            exposure[:]=0
            _,coverage=numerical_queries(corpus,[21,24],exposure)
            self.assertEqual(coverage['numerical_input_coverage'],0)


if __name__=='__main__':unittest.main()
