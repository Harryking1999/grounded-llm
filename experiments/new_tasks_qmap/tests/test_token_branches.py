"""Cycle correction keeps legitimate backtracking and every physical branch."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from grounded_llm.tasks.token_swapping.relation_data import TokenCorpus
from experiments.new_tasks_qmap.tests.test_exact_pairs import small_oracle
from experiments.new_tasks_qmap.src.relation_data import materialize,closure,token_goal_cohorts,sample_goals
from experiments.new_tasks_qmap.src.token_branches import expand,permitted,BranchReplay,cycle_rows,combine,first_uphill_rows


class BranchTests(unittest.TestCase):
    def test_first_uphill_includes_successful_detour_and_preserves_correct_returns(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(path)
            try:
                goal=0;start=int(np.flatnonzero(corpus.distances==3)[0])
                def up(s):return int(np.flatnonzero(corpus.distances[corpus.children[s]]>corpus.distances[s])[0])
                a=up(start);middle=int(corpus.children[start,a]);b=up(middle)
                near=int(corpus.children[goal,0]);detour=int(corpus.children[near,3]);other=int(corpus.children[detour,0])
                self.assertNotEqual(other,near);self.assertEqual(corpus.children[other,3],goal)
                def edge(q,s,choice):
                    rows=np.column_stack([s,np.zeros(len(s),dtype=np.int64)]);exact=materialize(corpus,rows)
                    return dict(query_index=np.array(q),source=np.array(s),goal=np.zeros(len(s),dtype=np.int64),
                        choice=np.array(choice),parent_distance=exact['distances'],child_distances=exact['child_distances'])
                records=[edge([0,1],[start,near],[a,3]),edge([0,1],[middle,detour],[b,0]),edge([1],[other],[3])]
                rows,stats=first_uphill_rows(records)
                np.testing.assert_array_equal(rows,np.unique([[start,0],[near,0]],axis=0))
                self.assertEqual(stats['first_uphill_queries'],2);self.assertEqual(stats['uphill_exchanges'],3)
                self.assertEqual(stats['solved_queries'],1);self.assertEqual(stats['extra_exchanges_on_success'],2)
                self.assertEqual(stats['verified_prefixes'],5)
                corrupt=deepcopy(records);corrupt[0]['child_distances'][0,a]=corrupt[0]['parent_distance'][0]
                with self.assertRaisesRegex(ValueError,'parity'):first_uphill_rows(corrupt)
            finally:corpus.close()

    def test_feedback_union_retains_old_branches_and_shared_fit_queries(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(path)
            try:
                data=dict(reserved_relations=np.array([],dtype=np.int64));partition=np.zeros(corpus.size,dtype=np.int8);banks=[]
                for root in [17,91]:
                    bank,_=expand(corpus,data,partition,np.array([[root,0]]),1,12)
                    bank.update(mining_queries=np.array([[root,0]]),raw_cycle_seeds=bank['tasks_seed'],switched_goals=np.empty((0,2),dtype=np.int64));banks.append(bank)
                banks[1]['raw_uphill_seeds']=np.array([[91,0]])
                merged,summary=combine(corpus,data,partition,banks,1,12)
                np.testing.assert_array_equal(merged['raw_uphill_seeds'],[[91,0]])
                rows=set(map(tuple,merged['tasks_branches']))
                for bank in banks:self.assertTrue(set(map(tuple,bank['tasks_branches'])).issubset(rows))
                replay=BranchReplay(corpus,data,partition,merged,dict(kind='branches',decision_fraction=.5,point_count=6),np.random.default_rng(1),fit_bank=banks[0])
                np.testing.assert_array_equal(replay.fit['tasks_branches_fit'],banks[0]['tasks_branches'])
                self.assertEqual(summary['teacher_paths_reached_goal'],summary['teacher_paths'])
                blocked=dict(reserved_relations=closure(corpus,banks[0]['tasks_seed']))
                with self.assertRaises(ValueError):combine(corpus,blocked,partition,banks,1,12)
            finally:corpus.close()

    def test_retry_optimization_preserves_checkpointed_proposal_stream(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,5,[(0,1),(1,2),(1,3),(3,4)]);corpus=TokenCorpus(path)
            try:
                reserved=closure(corpus,np.array([[7,8],[9,10],[11,12],[13,14]]));rng=np.random.default_rng(731)
                rows=sample_goals(corpus,np.arange(32),rng,reserved,ranking=True,allowed=np.arange(corpus.size),local_fraction=.5,walk_max=8)
                np.testing.assert_array_equal(rows[:,1],[42,42,39,119,21,116,12,45,85,118,46,53,86,18,102,26,90,11,55,115,5,104,5,53,38,80,36,23,34,119,13,38])
                np.testing.assert_array_equal(rng.integers(1000000,size=5),[274237,667837,680869,97371,357044])
                self.assertFalse(np.intersect1d(closure(corpus,rows),reserved).size)
            finally:corpus.close()

    def test_query_goal_holdout_allows_complete_paths_without_goal_label_leaks(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,6,[(0,1),(1,2),(1,3),(3,4),(4,5)]);corpus=TokenCorpus(path)
            try:
                config=dict(relations=dict(evaluation_count=3,training_goals=12,heldout_goals_per_split=2))
                groups,pairs,tasks,reserved,partition=token_goal_cohorts(corpus,config,np.random.default_rng(7))
                self.assertFalse(partition.any())
                self.assertTrue(np.isin(pairs[:,1],corpus.training_goal_pool).all())
                self.assertTrue(np.isin(tasks[:,1],corpus.training_goal_pool).all())
                self.assertEqual(len(pairs),corpus.size);self.assertEqual(len(tasks),corpus.size)
                self.assertFalse(np.intersect1d(closure(corpus,tasks),reserved).size)
                for name in ('unseen_goal_validation','unseen_goal_test'):
                    self.assertFalse(np.isin(groups[name][:,1],corpus.training_goal_pool).any())
                data=dict(reserved_relations=reserved,training_goal_pool=corpus.training_goal_pool)
                self.assertFalse(permitted(corpus,data,partition,groups['unseen_goal_test']).any())
                bank,summary=expand(corpus,data,partition,tasks[:30],1,20)
                self.assertGreater(summary['teacher_paths_reached_goal'],0)
                self.assertTrue(permitted(corpus,data,partition,bank['tasks_branches']).all())
            finally:corpus.close()

    def test_long_relation_holdout_keeps_terminal_neighborhood_trainable(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);small_oracle(path,6,[(0,1),(1,2),(1,3),(3,4),(4,5)]);corpus=TokenCorpus(path)
            try:
                config=dict(relations=dict(evaluation_count=8,training_goals=12,heldout_goals_per_split=2,minimum_relation_holdout_distance=5,preserve_training_goal_neighborhood=True))
                groups,_,_,reserved,partition=token_goal_cohorts(corpus,config,np.random.default_rng(31))
                for name in ('validation','test'):
                    self.assertTrue((corpus.distance(groups[name][:,0],groups[name][:,1])>=5).all())
                goals=corpus.training_goal_pool
                terminal_edges=np.column_stack([corpus.children[goals].ravel(),np.repeat(goals,corpus.max_candidates)])
                data=dict(reserved_relations=reserved,training_goal_pool=goals)
                self.assertTrue(permitted(corpus,data,partition,terminal_edges).all())
            finally:corpus.close()

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
