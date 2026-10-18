from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.blocks_distance_map.src.clear_distance_index import build_component, ClearDistanceIndex
from experiments.blocks_distance_map.src.symbolic_graph import build_symbolic, SymbolicComponent
from experiments.blocks_distance_map.src.full_graph_data import local_tiles
from experiments.blocks_distance_map.src.oracle import DistanceOracle


class ClearDistanceTest(unittest.TestCase):
    def test_every_indexed_state_and_distance_equal_independent_exact_oracle(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp); cells=list(range(10));tiles=local_tiles(cells)
            build_symbolic(len(cells),tiles,path/'graph',100000)
            graph=SymbolicComponent(path/'graph')
            build_component(path/'graph',path/'clear')
            index=ClearDistanceIndex(path/'clear');oracle=DistanceOracle()
            expected={int(s):oracle.distance(int(s),0) for s in graph.states[np.arange(graph.size)]}
            seen=set()
            for d,view in enumerate(index.views):
                masks=view.unrank(np.arange(view.size))
                self.assertEqual(set(map(int,masks)),{s for s,v in expected.items() if v==d})
                self.assertTrue(seen.isdisjoint(map(int,masks)));seen.update(map(int,masks))
                if view.size:
                    for s in index.sample(d,20,np.random.default_rng(d)):
                        self.assertEqual(oracle.distance(int(s),0),d)
            self.assertEqual(seen,{s for s,v in expected.items() if v>=0})


if __name__=='__main__':unittest.main()
