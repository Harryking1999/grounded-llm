"""Numerical deduplication must preserve complete arithmetic continuations."""
from contextlib import ExitStack,redirect_stdout
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from itertools import combinations
from unittest.mock import patch
from io import StringIO
import json
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.physical_data import save_scene
from grounded_llm.tasks.game24.environment import Game24
from grounded_llm.tasks.game24.relation_data import GameScene,GameCorpus
from grounded_llm.core.qmap.model import StateEncoder
from experiments.new_tasks_qmap.src.game_finite_evaluate import numerical_queries,main


class FiniteGameTests(unittest.TestCase):
    def test_deduplication_keeps_exact_reachability_and_both_targets(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            env=Game24(4);scenes=[];records=[];alphabet=[1,2,3,4,6]
            root=Path(directory);graph_root=root/'run/data/graphs';graph_root.mkdir(parents=True)
            for i,numbers in enumerate(combinations(alphabet,4)):
                graph=env.graph(numbers)
                values=[tuple(Fraction(v) for v in row) for row in graph['metadata']['exact_rational_states']]
                children=graph['children']
                for target in (21,24):
                    if (target,) not in values:
                        values.append((Fraction(target),));children=np.vstack([children,np.full((1,len(env.actions)),-1,dtype=np.int32)])
                goals=[values.index((t,)) for t in (21,24)]+[next(j for j,v in enumerate(values) if len(v)==n) for n in (2,3)]
                path=graph_root/str(i)
                record=save_scene(path,np.arange(len(values))[:,None],children,goals,
                    dict(task='game24',split='train',max_numbers=4,training_goals=2,original_state=0,source=dict(numbers=numbers)),state_dtype=np.int32)
                atomic_json(path/'values.json',[[str(v) for v in row] for row in values])
                records.append(dict(path=str(i),**record))
                scene=GameScene(path,record);scenes.append(scene)
                for name in ('states','children','distances','reverse_distances'):stack.callback(getattr(scene,name)._mmap.close)
            corpus=GameCorpus(scenes);exposure=np.ones(corpus.size,dtype=np.int64)
            groups,coverage=numerical_queries(corpus,[21,24],exposure)
            self.assertEqual(coverage['numerical_input_coverage'],1)
            self.assertLess(coverage['exact_numerical_states'],corpus.size)
            for target in (21,24):
                self.assertEqual(len(groups[f'root4_to_{target}']),len(scenes))
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
            # Exercise the real CLI lifecycle as well as query construction:
            # all cohorts, exact oracle, writing a negative result and cleanup.
            atomic_json(graph_root/'corpus.json',dict(scenes=records))
            held=np.array([[int(corpus.offsets[sid]),int(corpus.offsets[sid])+int(scene.goals[j])]
                for sid,scene in enumerate(scenes) for j in (2,3)],dtype=np.int64)
            np.savez(root/'run/data/relations.npz',tasks_unseen_goal_test=held)
            config=dict(task='game24',seed=1,data=dict(initial_sampling='exhaustive_distinct',initial_numbers=4,source_values=alphabet),
                model=dict(dimension=4,hidden=8,cap=16,metric='directed_max'),
                training=dict(cpu_threads=1,rank_temperature=1,unreachable_target=12),evaluation={})
            model=StateEncoder(input_dim=16,**config['model'])
            torch.save(dict(config=config,input_dim=16,model=model.state_dict(),step=1,source_commit='fixture',exposure=exposure),root/'run/best.pt')
            atomic_json(root/'run/summary.json',dict(evaluated_step=1,metrics=dict(reachability_threshold=6)))
            atomic_json(root/'protocol.json',dict(source_values=alphabet,targets=[21,24],expected_queries={k:len(v) for k,v in groups.items()},
                minimum_completion_rate=.8,target_completion_rate=.9,minimum_unreachable_recall=.8,maximum_finite_false_positive=.2))
            with patch('sys.argv',['evaluate','--protocol',str(root/'protocol.json'),'--run',str(root/'run'),'--out',str(root/'result.json')]),redirect_stdout(StringIO()):main()
            result=json.loads((root/'result.json').read_text())
            self.assertTrue(result['oracle_ok']);self.assertFalse(result['passed'])
            self.assertEqual(result['coverage']['numerical_input_coverage'],0)
            for cohort in result['metrics']['cohorts'].values():
                self.assertEqual(cohort['planning']['oracle']['count'],cohort['finite_count'])


if __name__=='__main__':unittest.main()
