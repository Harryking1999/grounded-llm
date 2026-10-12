from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader,DistributedSampler

from experiments.llm_task_sft.src import continuation
from experiments.llm_task_sft.src.epoch_queue import pending_jobs
from experiments.llm_task_sft.src.train import resume_position,scheduler_factor,validate_resume


def config():
    return json.loads((Path(__file__).parents[1]/'configs/blocks_reason_bs4.json').read_text())


def test_resume_allows_host_and_eval_changes_but_preserves_training_recipe():
    old=config();new=deepcopy(old)
    new['execution']['evaluation_ssh_port']=35016
    new['evaluation']['epoch_split']='test'
    validate_resume(dict(config=old,steps_per_epoch=12500),new,12500)
    new['training']['learning_rate']=1e-4
    with pytest.raises(ValueError):
        validate_resume(dict(config=old,steps_per_epoch=12500),new,12500)


def test_half_epoch_resume_preserves_each_ranks_order_and_model_rng():
    epoch,offset=resume_position(35,10)
    assert (epoch,offset)==(3,5)
    combined=[]
    for rank in range(4):
        sampler=DistributedSampler(range(40),num_replicas=4,rank=rank,shuffle=True,seed=20261011)
        sampler.set_epoch(epoch)
        loader=DataLoader(range(40),batch_size=1,sampler=sampler,
                          generator=torch.Generator().manual_seed(20261011+rank))
        before=torch.get_rng_state().clone()
        uninterrupted=[batch.item() for batch in loader]
        assert torch.equal(before,torch.get_rng_state())
        first=uninterrupted[:offset]
        resumed=[batch.item() for i,batch in enumerate(loader) if i>=offset]
        assert first+resumed==uninterrupted
        combined.extend(first+resumed)
    assert sorted(combined)==list(range(40))
    assert resume_position(40,10)==(4,0)


def test_restored_terminal_scheduler_remains_at_two_micro_without_new_warmup():
    parameter=torch.nn.Parameter(torch.zeros(1))
    optimizer=torch.optim.AdamW([parameter],lr=2e-5)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:scheduler_factor(s,37500,1875,0.1))
    saved=scheduler.state_dict();saved.update(last_epoch=37500,_last_lr=[2e-6])
    optimizer.param_groups[0]['lr']=2e-6
    scheduler.load_state_dict(saved)
    for _ in range(3):
        parameter.grad=torch.ones_like(parameter)
        optimizer.step();scheduler.step()
        assert scheduler.get_last_lr()[0]==pytest.approx(2e-6)
    assert scheduler_factor(43750,37500,1875,0.1)==pytest.approx(0.1)


def test_continuation_schedules_only_both_modes_of_510_test(tmp_path):
    contract=json.loads((Path(__file__).parents[1]/'configs/blocks_reason_continuation.json').read_text())
    assert continuation.cycle_steps(contract,config())==(12500,6250)
    jobs=pending_jobs(tmp_path,[('epoch_3.5',tmp_path/'checkpoint')],
                      {'validation':1100,'test':510},contract['evaluation']['modes'],25,
                      splits=contract['evaluation']['splits'])
    assert len(jobs)==42
    assert {job[2] for job in jobs}=={'test'}


def test_next_training_segment_follows_completed_evaluation(monkeypatch,tmp_path):
    events=[]
    def train(profile,config_path,config,checkpoint,target,out,status):
        events.append(('train',target))
        destination=out/'training'/f'checkpoint-{target}'
        destination.mkdir(parents=True)
        (destination/'evaluation_ready.json').write_text(json.dumps(dict(step=target,epoch=target/12500)))
        return destination
    def evaluate(profile,config_path,config,contract,checkpoint,out,status):
        target=json.loads((checkpoint/'evaluation_ready.json').read_text())['step']
        events.append(('evaluate',target))
        return dict(metrics={'rollout':{'arrival':{'rate':target/100000},
                                         'arrival_and_stop':{'rate':target/100000}}})
    monkeypatch.setattr(continuation,'train_segment',train)
    monkeypatch.setattr(continuation,'evaluate_checkpoint',evaluate)
    checkpoint=tmp_path/'original/checkpoint-37500'
    for target in (43750,50000):
        checkpoint=continuation.run_cycle({},None,config(),{},checkpoint,target,tmp_path,lambda *a,**k:None)
    assert events==[('train',43750),('evaluate',43750),('train',50000),('evaluate',50000)]
    assert json.loads((tmp_path/'curve.json').read_text())['best_observed_epoch']==4
