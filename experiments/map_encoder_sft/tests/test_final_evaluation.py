import copy
import json
from pathlib import Path

import pytest

from experiments.map_encoder_sft.src.final_evaluation import cache_evictions, completed, final_config
from experiments.map_sft_baselines.src.queue import evaluation_jobs

EXPERIMENT = Path(__file__).parents[1]


def configuration():
    base = json.loads((EXPERIMENT/'configs/five_task_50k.json').read_text())
    policy = json.loads((EXPERIMENT/'configs/final_evaluation.json').read_text())
    records = [dict(cohort=kind, trajectory_id=f'{kind}_{i}')
               for kind,n in policy['cohort_counts'].items() for i in range(n)]
    return base, policy, records


def test_final_jobs_cover_the_accepted_cohort_and_all_tasks(tmp_path):
    base,policy,records = configuration()
    original = copy.deepcopy(base)
    config = final_config(base,policy,records)
    assert base == original
    assert config['data']['tasks'] == base['generalization']['task_pool']
    assert config['map'] == base['map']
    assert config['training'] == base['training']
    jobs = evaluation_jobs(tmp_path,config,[dict(baseline='B4_paired_address_full',step=250000,epoch=20)])
    for mode in ('RO','TF'):
        ranges=[(job[4],job[5]) for job in jobs if job[3]==mode]
        assert [i for start,stop in ranges for i in range(start,stop)] == list(range(2000))


def test_final_rejects_missing_or_duplicate_queries():
    base,policy,records = configuration()
    with pytest.raises(ValueError):
        final_config(base,policy,records[:-1])
    records[-1]['trajectory_id'] = records[-2]['trajectory_id']
    with pytest.raises(ValueError):
        final_config(base,policy,records)


def test_endpoint_completion_requires_full_ro_and_tf(tmp_path):
    base,policy,records = configuration()
    config = final_config(base,policy,records)
    path=tmp_path/'evaluation/B4_paired_address_full/checkpoint-250000/summary.json'
    path.parent.mkdir(parents=True)
    row=dict(step=250000,metrics=dict(RO=dict(tasks=2000),TF=dict(tasks=1250)))
    path.write_text(json.dumps(row))
    assert not completed(tmp_path,'B4_paired_address_full',250000,config)
    row['metrics']['TF']['tasks']=2000
    path.write_text(json.dumps(row))
    assert completed(tmp_path,'B4_paired_address_full',250000,config)


def test_returned_remote_curve_confirms_endpoint_completion(tmp_path):
    base,policy,records = configuration()
    config = final_config(base,policy,records)
    path=tmp_path/'results/B4_paired_address_full/curve.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([dict(step=250000,metrics=dict(RO=dict(tasks=2000),TF=dict(tasks=2000)))]))
    assert completed(tmp_path,'B4_paired_address_full',250000,config)
    assert not completed(tmp_path,'B4_paired_address_full',200000,config)


def test_cache_reuses_finished_weights_and_evicts_only_archived_id_entries():
    old=dict(baseline='old',step=250000,size=4)
    pending=dict(baseline='pending',step=250000,size=4)
    selected=cache_evictions([old,pending],incoming_bytes=4,maximum_bytes=8,
        pending={('pending',250000)},archived={('old',250000),('pending',250000)})
    assert selected==[old]
    with pytest.raises(RuntimeError):
        cache_evictions([old,pending],4,8,{('pending',250000)},set())
    assert cache_evictions([pending],0,8,{('pending',250000)},set())==[]
