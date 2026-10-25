from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from grounded_llm.tasks.token_swapping.relation_data import TokenCorpus
from grounded_llm.tasks.token_swapping.environment import TokenSwapping
from grounded_llm.core.qmap.transitions import TransitionQMap
from experiments.new_tasks_qmap.tests.test_exact_pairs import small_oracle
from experiments.new_tasks_qmap.src.relation_data import closure,materialize
from experiments.new_tasks_qmap.src.token_branches import permitted
from experiments.new_tasks_qmap.src.token_profiles import build,AnchorProfiles
from experiments.new_tasks_qmap.src.relations import train


class ProfileTests(unittest.TestCase):
    def test_exact_targets_protected_coordinates_and_zero_masked_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'oracle').mkdir();small_oracle(root/'oracle',5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(root/'oracle')
            try:
                anchors=np.array([0,7,18,39]);partition=np.zeros(corpus.size,dtype=np.int8);partition[37]=1
                data=dict(training_goal_pool=anchors,reserved_relations=closure(corpus,np.array([[31,0]])))
                build(corpus,data,partition,root/'profiles',2)
                profiles=AnchorProfiles(root/'profiles',corpus,data,partition,4,'cpu')
                expected=corpus.distance(np.arange(corpus.size)[:,None],anchors[None,:])
                np.testing.assert_array_equal(profiles.targets,expected)
                a,b=data['reserved_relations']//corpus.size,data['reserved_relations']%corpus.size
                for source,goal in zip(a,b):
                    for x,y in [(source,goal),(goal,source)]:
                        if y in anchors:self.assertFalse(profiles.allowed[x,np.flatnonzero(anchors==y)[0]])
                self.assertFalse(profiles.allowed[37].any())
                vectors=torch.randn(corpus.size,4,requires_grad=True);loss=profiles.loss(vectors,np.arange(corpus.size));loss.backward()
                self.assertTrue((vectors.grad[~profiles.allowed]==0).all())
                profiles.targets[~profiles.allowed]=255
                self.assertEqual(float(profiles.loss(vectors,np.arange(corpus.size)).detach()),float(loss.detach()))
            finally:corpus.close()

    def test_training_resume_matches_and_reader_uses_no_teacher_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'oracle').mkdir();small_oracle(root/'oracle',5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(root/'oracle')
            try:
                partition=np.zeros(corpus.size,dtype=np.int8)
                data=dict(training_goal_pool=np.array([0,7,18,39]),reserved_relations=closure(corpus,np.array([[31,0]])))
                rows=np.column_stack([np.arange(1,corpus.size),np.zeros(corpus.size-1,dtype=np.int64)])
                rows=rows[permitted(corpus,data,partition,rows)][:16]
                data.update(pairs_train=rows,pair_distances_train=corpus.distance(rows[:,0],rows[:,1]))
                data.update({k+'_train':v for k,v in materialize(corpus,rows).items()})
                data.update({k+'_validation':v for k,v in materialize(corpus,np.array([[31,0]])).items()})
                build(corpus,data,partition,root/'profiles',2)
                cfg=dict(task='token_swapping',seed=41,model=dict(dimension=4,hidden=16,cap=32,metric='linf'),
                    training=dict(steps=4,batch_size=8,decision_batch_size=4,learning_rate=.001,rank_weight=1.,rank_temperature=1.,eval_every=2,cpu_threads=1,unreachable_target=24.,evaluation_checkpoint='last.pt',anchor_profiles=dict(weight=1.,prepared_from=str(root/'profiles'))),
                    evaluation=dict(decision_count=4,rollout_count=2,horizon=24))
                def prepared(config,path):
                    path.mkdir(exist_ok=True);np.save(path/'endpoint_partition.npy',partition);return corpus,data,dict(finite_max=8)
                with patch('experiments.new_tasks_qmap.src.relations.prepare',side_effect=prepared):
                    train(cfg,root/'continuous','fixture','cpu')
                    short=deepcopy(cfg);short['training']['steps']=2;train(short,root/'short','fixture','cpu')
                    resumed=deepcopy(cfg);resumed['continuation']=dict(checkpoint=str(root/'short/last.pt'))
                    train(resumed,root/'resumed','fixture','cpu')
                a=torch.load(root/'continuous/last.pt',weights_only=False);b=torch.load(root/'resumed/last.pt',weights_only=False)
                for key,value in a['model'].items():torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
                np.testing.assert_array_equal(a['anchor_probe_ids'],b['anchor_probe_ids'])
                env=TokenSwapping(corpus.n,corpus.env.edges)
                with patch.object(AnchorProfiles,'__init__',side_effect=RuntimeError('No training cache at inference')),patch.object(TokenCorpus,'distance',side_effect=RuntimeError('No oracle at inference')):
                    reader=TransitionQMap(env,root/'continuous/last.pt','token_swapping')
                    reader.encode([tuple(range(5)),(1,0,2,3,4)])
            finally:corpus.close()


if __name__=='__main__':unittest.main()
