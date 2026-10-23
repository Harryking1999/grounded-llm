"""Bounded optimal-continuation closure with exact all-action labels."""
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from .solution_structure import load_corpus, permitted, context_points, exact_path
from .relation_data import materialize, keys


def continuation_closure(corpus, reserved, rows, limit, recover=False):
    """Reject an entire component on protection/size boundaries; never truncate."""
    goal=int(rows[0,1]); pending=set(map(int,rows[:,0])); seen=set()
    if recover:
        children=corpus.successors(rows[:,0]); candidates=np.unique(children[children>=0])
        labels=corpus.distance(candidates,np.full(len(candidates),goal))
        pending.update(map(int,candidates[np.isfinite(labels)&(labels>0)&(labels<=32)]))
    while pending:
        states=np.array(sorted(pending-seen-{goal}),dtype=np.int64)
        if not len(states): break
        contexts=np.column_stack([states,np.full(len(states),goal)])
        if len(seen)+len(states)>limit: return None,'size'
        if not permitted(corpus,reserved,contexts).all(): return None,'protected'
        truth=corpus.distance(states,contexts[:,1]);children=corpus.successors(states)
        valid=children>=0;labels=np.full(children.shape,np.inf)
        labels[valid]=corpus.distance(children[valid],np.full(valid.sum(),goal))
        if not (valid&(labels==truth[:,None]-1)).any(axis=1).all():
            raise ValueError('No exact optimal continuation')
        seen.update(map(int,states));pending=set(map(int,children[valid&(labels==truth[:,None]-1)]))-seen-{goal}
    return np.column_stack([sorted(seen),np.full(len(seen),goal)]).astype(np.int64),None


def prepare_branches(config,out):
    options=config['data'];source=Path(options['branch_source']).resolve()
    atomic_json(out/'scenes.json',json.loads((source/'scenes.json').read_text()))
    corpus=load_corpus(config,out);old=dict(np.load(source/'relations.npz'))
    reserved=old['reserved_relations'];base=[];expanded=[];rejected={};paths=[]
    for name in ['terminal_groups','auxiliary_groups']:
        for group in old[name]:
            rows=old['tasks_train'][group[group>=0]]
            closure,reason=continuation_closure(corpus,reserved,rows,options['closure_limit'],options['recover_one_step'])
            if closure is None:
                rejected[reason]=rejected.get(reason,0)+1;continue
            base.append(rows);expanded.append(closure)
            if name=='terminal_groups': paths.append(np.r_[rows[:,0],rows[0,1]])
    baseline=np.unique(np.concatenate(base),axis=0);closed=np.unique(np.concatenate(expanded),axis=0)
    contexts=baseline if options['branch_condition']=='path_control' else closed
    exact=materialize(corpus,contexts);pairs=np.unique(context_points(contexts,exact['children']),axis=0)
    data={k:v for k,v in old.items() if not k.endswith('_train') and k not in ['terminal_groups','auxiliary_groups','terminal_ids','auxiliary_ids']}
    data.update({k+'_train':v for k,v in exact.items()})
    data.update(pairs_train=pairs,pair_distances_train=corpus.distance(pairs[:,0],pairs[:,1]),
                terminal_ids=np.arange(len(contexts)),auxiliary_ids=np.array([],dtype=np.int64))
    # Identical accepted teacher starts in both conditions; fitting only.
    padded=np.full((len(paths),33),-1,dtype=np.int64)
    for i,path in enumerate(paths):padded[i,:len(path)]=path
    for k,v in materialize(corpus,np.array([[p[0],p[-1]] for p in paths])).items():data[k+'_fit_validation']=v
    data['solution_paths_fit_validation']=padded
    # Hold out whole intermediate target identities from BOTH candidate banks.
    union_pairs=np.unique(context_points(closed,corpus.successors(closed[:,0])),axis=0)
    train_keys=keys(union_pairs[:,0],union_pairs[:,1],corpus.size)
    used_goals=set(map(int,closed[:,1]));rng=np.random.default_rng(options['structure_seed'])
    for source_cohort,name in [('validation','new_goal_validation'),('unseen_layout_validation','new_layout_goal_validation')]:
        chosen=[];targets=set()
        for padded_path in old['solution_paths_'+source_cohort]:
            path=padded_path[padded_path>=0]
            if len(path)<4:continue
            sid=int(corpus.locate(path[0]));scene=corpus.scenes[sid];offset=corpus.offsets[sid]
            original_goals=set(map(int,scene.goals+offset))
            candidates=[int(t) for t in path[1:-1] if int(t) not in original_goals|used_goals|targets]
            if not candidates:continue
            target=candidates[len(candidates)//2]
            if sum(int(corpus.locate(t))==sid for t in targets)>=2:continue
            scene=corpus.scenes[sid];offset=corpus.offsets[sid]
            states=np.arange(len(scene.states))+offset
            labels=corpus.distance(states,np.full(len(states),target));count_before=len(chosen)
            for lo,hi in [(0,5),(5,10),(10,20),(20,32)]:
                for start in rng.permutation(states[(labels>lo)&(labels<=hi)]):
                    row=np.array([[start,target]]);cp=context_points(row,corpus.successors(row[:,0]))
                    if np.intersect1d(train_keys,keys(cp[:,0],cp[:,1],corpus.size)).size:continue
                    chosen.append(exact_path(corpus,int(start),target,32));break
            if len(chosen)>count_before:targets.add(target)
        if not chosen:raise ValueError('No held-out intermediate target queries')
        for k,v in materialize(corpus,np.array([[p[0],p[-1]] for p in chosen])).items():data[k+'_'+name]=v
        array=np.full((len(chosen),33),-1,dtype=np.int64)
        for i,p in enumerate(chosen):array[i,:len(p)]=p
        data['solution_paths_'+name]=array
    np.savez_compressed(out/'relations.npz',**data)
    summary=dict(task='sokoban',finite_max=float(exact['distances'].max()),baseline_contexts=len(baseline),
        closed_contexts=len(closed),selected_contexts=len(contexts),point_pairs=len(pairs),accepted_components=len(base),
        accepted_terminal_paths=len(paths),rejected=rejected,all_candidates_preserved=True,
        closure='Every accepted optimal continuation reaches its target; optional one-step finite detours also get optimal continuation. No global closure of all detours claimed.',
        new_goal_counts={n:len(data['tasks_'+n]) for n in ['new_goal_validation','new_layout_goal_validation']})
    atomic_json(out/'summary.json',summary)
    return corpus,data,summary
