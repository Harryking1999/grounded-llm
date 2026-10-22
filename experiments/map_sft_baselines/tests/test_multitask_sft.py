import numpy as np
import torch
from grounded_llm.tasks.path.environment import action_catalog
from grounded_llm.tasks.path.sft import Environment, demonstration
from grounded_llm.tasks.restoration_sft import make_step
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.restoration_sft import demonstration as restoration_demo
from experiments.map_sft_baselines.src.multitask_prepare import path_records


class Reader:
    def encode(self, ids):
        return torch.tensor([[float(i), 0.] for i in ids])

    def distance(self, a, b):
        return (a-b).abs().sum(-1)


def test_path_renumber_keeps_moves_vectors_and_history_aligned(tmp_path):
    adjacency=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=bool)
    actions,_=action_catalog(adjacency)
    np.savez(tmp_path/'graph.npz',adjacency=adjacency,actions=actions)
    env=Environment(tmp_path/'graph.npz')
    steps=[make_step(env,Reader(),s,2) for s in [0,1,2]]
    record=dict(start=0,goal=2,reference_actions=['move(0,1)','move(1,2)'])
    for with_map in [True,False]:
        demo=demonstration(env,record,steps,np.random.default_rng(8),with_map)
        assert demo.executed_path==(0,1,2)
        assert 'Actual executed path: [0, 1]' in demo.turns[1].user_text
        for turn in demo.turns:
            assert '{' not in turn.user_text
            if turn.chosen_id is not None:
                action,_=turn.step.execute(turn.chosen_id)
                assert f'{turn.chosen_id}: {action}' in turn.user_text
            assert ('Map-distance ranking' in turn.answer_text)==(with_map and turn.chosen_id is not None)


def test_path_sampling_excludes_heldout_decisions_on_entire_route(tmp_path):
    adjacency=np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],dtype=bool)
    actions,_=action_catalog(adjacency)
    np.savez(tmp_path/'graph.npz',adjacency=adjacency,actions=actions)
    env=Environment(tmp_path/'graph.npz')
    rows,covered,_,_=path_records(env,Reader(),3,{(1,3)},11)
    assert len(rows)==3 and (1,3) not in covered
    for r in rows:
        assert r['shortest_moves']==env.distance(r['start'],r['goal'])
        assert all(pair!=(1,3) for pair in r['pairs'])


def test_restoration_plain_keeps_physical_supervision_without_ranking():
    env=Cube2(metric='QTM_URF_6');start=env.step(0,'U')
    steps=[make_step(env,Reader(),s) for s in [start,0]]
    record=dict(start=start,goal=0,reference_actions=["U'"])
    mapped=restoration_demo('cube2',env,record,steps)
    plain=restoration_demo('cube2',env,record,steps,with_map=False)
    assert mapped.executed_path==plain.executed_path
    assert [t.chosen_id for t in mapped.turns]==[t.chosen_id for t in plain.turns]
    assert all('Map-distance ranking' not in t.answer_text for t in plain.turns)
    assert 'First output "Map-distance ranking' not in plain.turns[0].user_text
    assert plain.turns[-1].answer_text.endswith('<done/>')
