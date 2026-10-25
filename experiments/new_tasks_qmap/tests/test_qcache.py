import unittest
from types import SimpleNamespace
import numpy as np
import torch
from grounded_llm.core.qmap.model import StateEncoder
from experiments.new_tasks_qmap.src.relation_evaluate import QCache


class QCacheTests(unittest.TestCase):
    def test_table_preserves_repeated_order_scores_and_lazy_feature_reads(self):
        torch.manual_seed(17);features=np.random.default_rng(9).normal(size=(32,8)).astype(np.float32)
        seen=[]
        def read(ids):
            seen.append(ids.copy());return features[ids]
        corpus=SimpleNamespace(size=32,features=read)
        model=StateEncoder(16,12,32,'linf',input_dim=8)
        dense=QCache(model,corpus,'cpu');dictionary=QCache(model,corpus,'cpu',max_table_bytes=0)
        for ids in [np.array([[7,3,7],[1,3,2]]),np.array([[9,7],[9,1]])]:
            torch.testing.assert_close(dense.q(ids),dictionary.q(ids),rtol=0,atol=0)
        np.testing.assert_array_equal(seen[2],[9])
        a,b=np.array([7,9,1]),np.array([3,7,2])
        np.testing.assert_array_equal(dense.scores(a,b),dictionary.scores(a,b))
        self.assertEqual(int(dense.present.sum()),5)


if __name__=='__main__':unittest.main()
