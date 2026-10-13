from experiments.blocks_distance_map.src.oracle import DistanceOracle
from experiments.gcml_counterexamples.src import blocks as rules
from experiments.llm_task_sft.src.analyze_policies import (
    compare, first_failure, normalize, physical_action, summarize)


def test_first_fatal_move_is_not_replaced_by_later_dead_state_stop():
    goal = (1<<98) | (1<<99)
    start = goal | 15
    move = {'shape_id':5,'row':0,'col':1}
    after = rules.apply(start,move)
    raw = dict(trajectory_id='a',executed_actions=[move],arrived=False,
        arrived_and_stopped=False,failure='premature_stop',turns=[
            dict(index=0,before_state=str(start),before_distance=2,action=move,
                 reachability=False,shortest_progress=False,control_error=None),
            dict(index=1,before_state=str(after),before_distance=-1,
                 reachability=False,control_error='premature_stop')])
    record = dict(start=str(start),goal=str(goal),group='new_initial_board',shortest_moves=2)
    case = normalize(raw,record,'sft')
    failure = first_failure(case,DistanceOracle())
    assert failure == dict(step=1,kind='untileable_residue',detail='unsupported_cell')
    summary,_ = summarize({'a':case},DistanceOracle())
    assert summary['by_step']['1']['first_fatal']==1
    assert summary['by_step']['2']['first_fatal']==0
    assert summary['by_step']['2']['already_dead_turns']==1


def test_goal_destruction_and_live_stop_are_distinct_first_errors():
    goal = (1<<98) | (1<<99)
    start = goal | 15
    common = dict(id='a',goal=goal,arrived=False,rows=[])
    action = (5,9,8)
    destroyed = dict(common,rows=[dict(step=1,live=True,before=start,remaining=2,
        action=action,after=rules.apply(start,dict(zip(('shape_id','row','col'),action))),safe=False,error=None)])
    assert first_failure(destroyed,DistanceOracle())['kind']=='remove_goal_cells'
    stopped = dict(common,rows=[dict(step=1,live=True,before=start,remaining=2,
        action=None,after=start,safe=False,error='premature_stop')])
    assert first_failure(stopped,DistanceOracle())['kind']=='premature_stop'


def test_compare_uses_physical_actions_and_separates_success_sets():
    move = {'shape_id':5,'row':0,'col':0}
    index = rules.BY_ACTION[(5,0,0)]
    assert physical_action(index)==physical_action(move)
    def cases(successes):
        return {k:dict(id=k,start=15,goal=0,length=2,arrived=k in successes,
                       moves=[physical_action(move)]) for k in ('a','b','c')}
    left,right=cases({'a','b'}),cases({'b','c'})
    fatal=lambda values:{k:None if v['arrived'] else dict(step=2,kind='premature_stop') for k,v in values.items()}
    result=compare(left,right,fatal(left),fatal(right))
    assert (result['both_arrive'],result['left_only'],result['right_only'],result['neither'])==(1,1,1,0)
    assert result['identical_first_physical_action']['numerator']==3
    assert result['success_set_jaccard']==1/3
    assert result['common_success_path_lengths']['tasks']==1
    assert result['common_success_path_lengths']['left_mean_moves']==1
