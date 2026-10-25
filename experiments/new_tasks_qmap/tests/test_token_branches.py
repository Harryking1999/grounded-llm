"""Cycle correction keeps legitimate backtracking and every physical branch."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from grounded_llm.tasks.token_swapping.relation_data import TokenCorpus
from experiments.new_tasks_qmap.tests.test_exact_pairs import small_oracle
from experiments.new_tasks_qmap.src.relation_data import materialize,closure
from experiments.new_tasks_qmap.src.token_branches import expand,permitted,BranchReplay,cycle_rows


class BranchTests(unittest.TestCase):
    def test_both_cycle_ends_keep_correct_backtrack_and_teacher_continuation(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(path)
            try:
                a=int(corpus.children[0,0]);b=int(corpus.children[a,1]);roots=np.array([[a,0],[b,0]])
                partition=np.zeros(corpus.size,dtype=np.int8);data=dict(reserved_relations=np.array([],dtype=np.int64))
                bank,summary=expand(corpus,data,partition,roots,2,12)
                self.assertTrue(summary['teacher_paths_reached_goal']==summary['teacher_paths'])
                exact=materialize(corpus,roots)
                self.assertFalse(exact['optimal'][0,np.flatnonzero(exact['children'][0]==b)[0]])
                self.assertTrue(exact['optimal'][1,np.flatnonzero(exact['children'][1]==a)[0]])
                self.assertGreater(len(bank['tasks_branches']),len(bank['tasks_seed']))
                np.testing.assert_array_equal(materialize(corpus,bank['tasks_branches'])['children'],corpus.successors(bank['tasks_branches'][:,0]))
            finally:corpus.close()

    def test_protected_context_rejected_whole_without_trimming_a_candidate(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(path)
            try:
                roots=np.column_stack([np.arange(1,corpus.size),np.zeros(corpus.size-1,dtype=np.int64)])
                data=dict(reserved_relations=closure(corpus,roots[[4]]));partition=np.zeros(corpus.size,dtype=np.int8);partition[13]=1
                bank,summary=expand(corpus,data,partition,roots,2,12)
                self.assertGreater(summary['rejected_seeds'],0)
                self.assertTrue(permitted(corpus,data,partition,bank['tasks_branches']).all())
                self.assertFalse(np.intersect1d(closure(corpus,bank['tasks_branches']),data['reserved_relations']).size)
                self.assertEqual(corpus.successors(bank['tasks_branches'][:,0]).shape[1],4)
            finally:corpus.close()

    def test_replay_preserves_broad_pairs_and_restores_its_random_state(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(path)
            try:
                a=int(corpus.children[0,0]);b=int(corpus.children[a,1]);roots=np.array([[a,0],[b,0]])
                data=dict(reserved_relations=np.array([],dtype=np.int64));partition=np.zeros(corpus.size,dtype=np.int8)
                bank,_=expand(corpus,data,partition,roots,1,10)
                replay=BranchReplay(corpus,data,partition,bank,dict(kind='branches',decision_fraction=.5,point_count=6),np.random.default_rng(1))
                pair=roots.copy();tasks=np.tile(roots,(2,1));truth=corpus.distance(pair[:,0],pair[:,1])
                args=(pair,np.arange(2),truth,truth,tasks,np.arange(4),np.zeros(2,dtype=np.uint8),np.zeros(4,dtype=np.uint8))
                saved=deepcopy(replay.state_dict());first=replay.apply(*[v.copy() for v in args])
                np.testing.assert_array_equal(first[0][:2],pair);np.testing.assert_array_equal(first[1][:2],np.arange(2))
                self.assertEqual(len(first[0]),8)
                replay.load_state_dict(saved);second=replay.apply(*[v.copy() for v in args])
                for x,y in zip(first,second):np.testing.assert_array_equal(x,y)
            finally:corpus.close()

    def test_extracts_cycle_members_and_excludes_prefix(self):
        records=[]
        for source,target,repeated in [(4,5,False),(5,6,False),(6,5,True)]:
            records.append(dict(query_index=np.array([0]),source=np.array([source]),goal=np.array([0]),
                children=np.array([[target]]),choice=np.array([0]),next_repeated=np.array([repeated])))
        rows,lengths=cycle_rows(records)
        np.testing.assert_array_equal(rows,[[5,0],[6,0]]);self.assertEqual(lengths,[2])


if __name__=='__main__':unittest.main()
