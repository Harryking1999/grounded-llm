from experiments.flamingo_map_reader.src.probe_map_content import aggregate_pairs, select_records


def test_selection_is_fixed_and_independent_of_model_outcomes():
    records=[dict(trajectory_id=f'{origin}_{length}_{i}',group=origin,shortest_moves=length)
             for origin in ('new','old') for length in (6,7) for i in range(5)]
    config=dict(seed=13,tasks_per_origin_by_shortest_length={'6':2,'7':3},sample_tasks=10)
    a=select_records(records,config)
    b=select_records(list(reversed(records)),config)
    assert [r['trajectory_id'] for r in a]==[r['trajectory_id'] for r in b]
    assert len({r['trajectory_id'] for r in a})==10


def test_paired_actions_and_presented_map_minimum_are_separate():
    score=lambda safe,minimum:dict(action_keeps_goal_reachable=safe,
        action_environment_shortest=safe,action_map_minimum=minimum)
    row=dict(stage='first',normal=score(True,True),reverse=score(False,False),
        changed_physical_action=True,reverse_presented_map_minimum=True,
        normal_saved_action_agreement=False,distance_order_changed=True)
    result=aggregate_pairs([dict(group='new',shortest_moves=6,turns=[row])])['all']
    assert result['normal_safe']==1 and result['reverse_safe']==0
    assert result['paired_safe_counts']=={'safe/unsafe':1}
    assert result['reverse_original_map_minimum']==0
    assert result['reverse_presented_map_minimum']==1
