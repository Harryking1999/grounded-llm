import copy
import json
from pathlib import Path
import pytest
from torch.utils.data import DistributedSampler
from grounded_llm.core.sft.schedule import validate_resume_contract

BASE='B4_joint_content_frozen'

def configs():
    new=json.loads((Path(__file__).parents[1]/'configs/five_task_50k.json').read_text())
    old=copy.deepcopy(new)
    old['baselines'][BASE].pop('training')
    saved=dict(baseline=BASE,config=old,step=100000)
    return saved,new

def test_topology_requires_explicit_permission_and_preserves_other_conditions():
    saved,new=configs()
    with pytest.raises(ValueError):validate_resume_contract(saved,new,BASE)
    validate_resume_contract(saved,new,BASE,allow_topology_change=True)
    for key,value in [('learning_rate',1e-3),('micro_batch',2),('global_batch',8),('epochs',21)]:
        changed=copy.deepcopy(new);changed['training'][key]=value
        with pytest.raises(ValueError):validate_resume_contract(saved,changed,BASE,allow_topology_change=True)
    full=copy.deepcopy(saved);full['config']['baselines'][BASE]['full_llm']=True
    with pytest.raises(ValueError):validate_resume_contract(full,new,BASE,allow_topology_change=True)

def test_single_rank_accumulation_preserves_global_update_sample_sets():
    data=list(range(40))
    for epoch in (0,7):
        samplers=[DistributedSampler(data,num_replicas=2,rank=r,seed=23) for r in range(2)]
        single=DistributedSampler(data,num_replicas=1,rank=0,seed=23)
        for sampler in [*samplers,single]:sampler.set_epoch(epoch)
        ranks=[list(s) for s in samplers];one=list(single)
        for update in range(10):
            assert set(one[4*update:4*update+4])==set(ranks[0][2*update:2*update+2]+ranks[1][2*update:2*update+2])
