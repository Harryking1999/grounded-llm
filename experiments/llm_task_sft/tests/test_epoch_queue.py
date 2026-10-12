from experiments.llm_task_sft.src.epoch_queue import pending_jobs


def test_each_epoch_includes_both_modes_on_the_full_510_test_tasks(tmp_path):
    models=[('base',tmp_path/'base'),('epoch_1',tmp_path/'epoch_1'),('epoch_2',tmp_path/'epoch_2')]
    counts={'validation':1100,'test':510}
    jobs=pending_jobs(tmp_path,models,counts,['rollout','reference'],25)
    for label in ['base','epoch_1','epoch_2']:
        for mode in ['rollout','reference']:
            shards=[job[4] for job in jobs if job[0]==label and job[2]=='test' and job[3]==mode]
            assert shards==list(range(21))
    complete=tmp_path/'epoch_1/test/rollout/shard_000/summary.json'
    complete.parent.mkdir(parents=True);complete.write_text('{}')
    resumed=pending_jobs(tmp_path,models,counts,['rollout','reference'],25)
    assert len(resumed)==len(jobs)-1
    assert all(job[-1]!=complete.parent for job in resumed)
    assert all(job[0]!='selected_test' for job in resumed)
