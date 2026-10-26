"""Freeze the original Rush Hour query suite when appending training layouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from .relation_data import load,materialize,LocalPathSampler,training_decision_mask


def freeze_queries(corpus,expanded,reference):
    result=dict(expanded);names=[key[6:] for key in reference if key.startswith('tasks_') and key!='tasks_train'];counts={}
    for name in names:
        rows=reference['tasks_'+name];fields=materialize(corpus,rows)
        for key,value in fields.items():
            old=reference[key+'_'+name]
            # Appended scenes may have more action slots; old legal actions
            # and their order must remain exact, including unreachable ones.
            if value.ndim==2 and key in ('children','child_distances','optimal'):
                if value.shape[1]<old.shape[1]:raise ValueError('Old candidate width shrank')
                np.testing.assert_array_equal(value[:,:old.shape[1]],old,err_msg=key+'_'+name)
                if key=='children' and (value[:,old.shape[1]:]>=0).any():raise ValueError('Old query gained a physical action')
            else:np.testing.assert_array_equal(value,old,err_msg=key+'_'+name)
            result[key+'_'+name]=value
        counts[name]=len(rows)
    return result,counts


def pad_replay(bank,width):
    result={key:np.asarray(value).copy() for key,value in bank.items()}
    for key,fill in [('children',-1),('child_distances',np.inf),('optimal',False),
                     ('path_children',-1),('path_child_distances',np.inf),('path_optimal',False)]:
        value=result[key]
        if value.shape[-1]>width:raise ValueError('Replay candidate width exceeds the new corpus')
        padding=[(0,0)]*value.ndim;padding[-1]=(0,width-value.shape[-1])
        result[key]=np.pad(value,padding,constant_values=fill)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());reference=(args.run/cfg['data']['freeze_evaluation_from']).resolve()
    previous=json.loads((reference.parent/'config.json').read_text())
    old,old_data,_=load(dict(previous,input_encoding=cfg['input_encoding']),reference)
    corpus,data,description=load(cfg,args.run/'data')
    if len(corpus.scenes)<=len(old.scenes):raise ValueError('This preparation must append new layouts')
    np.testing.assert_array_equal(corpus.offsets[:len(old.scenes)+1],old.offsets)
    for a,b in zip(old.scenes,corpus.scenes):
        if a.metadata!=b.metadata:raise ValueError('Old scene metadata changed')
        for field in ('states','children','goals'):np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    result,counts=freeze_queries(corpus,data,old_data)
    for name in counts:
        if training_decision_mask(corpus,result,result['tasks_'+name]).any():
            raise ValueError('A frozen evaluation decision is allowed by the expanded training boundaries: '+name)
    archive=args.run/'data/relations_unfrozen.npz'
    if archive.exists():raise FileExistsError('Preserve the original expanded relation archive')
    (args.run/'data/relations.npz').rename(archive)
    np.savez_compressed(args.run/'data/relations.npz',**result)
    description['unfrozen_expanded_cohorts']=description['cohorts']
    description['cohorts']={name:dict(count=count,reachable=int(np.isfinite(result['distances_'+name]).sum())) for name,count in counts.items()}
    atomic_json(args.run/'data/summary.json',description)
    spec=cfg['data']['scale_replay'];source=(args.run/spec['source']).resolve();target=(args.run/spec['target']).resolve()
    with np.load(source) as value:bank=pad_replay(dict(value),corpus.max_candidates)
    target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**bank)
    sampler=LocalPathSampler(corpus,result,bank,cfg['training']['failure_replay'],np.random.default_rng(781),.75)
    train={s.metadata['catalogue_group'] for s in corpus.scenes if s.metadata['split']=='train'}
    held={s.metadata['catalogue_group'] for s in corpus.scenes if s.metadata['split']!='train'}
    if train&held:raise ValueError('Whole-layout holdout leaked into training')
    output=dict(passed=True,training_catalogues=len(train),old_scene_prefix=len(old.scenes),old_state_prefix=old.size,
        states=corpus.size,max_candidates=corpus.max_candidates,frozen_query_counts=counts,old_physics_and_goals_identical=True,
        whole_layout_overlap=0,replay=sampler.summary,
        scope='Append-only layout scaling; same frozen old queries, exact labels and physical actions. Candidate padding adds no legal action. New-layout internal holdouts remain protected in the preserved full archive; checkpoint selection and primary tests use only the old suite.')
    atomic_json(args.out,output);print(json.dumps(output),flush=True)


if __name__=='__main__':main()
