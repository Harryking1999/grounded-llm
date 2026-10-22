"""Prepare exact state-pair cohorts while excluding held-out relation closures."""
import importlib
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json


def keys(source, goal, size):
    source, goal = np.broadcast_arrays(source, goal)
    return np.minimum(source, goal)*np.int64(size)+np.maximum(source, goal)


def closure(corpus, rows):
    children=corpus.successors(rows[:,0]); valid=children>=0
    return np.unique(np.concatenate([keys(rows[:,0],rows[:,1],corpus.size),
        keys(children[valid],np.broadcast_to(rows[:,1,None],children.shape)[valid],corpus.size)]))


def balanced(pool, count, distances, rng):
    """Half broad proposals, half uniform true-distance layers; exact labels only."""
    if not len(pool): raise ValueError('No eligible exact relations')
    broad=rng.choice(len(pool),count//2,replace=len(pool)<count//2)
    levels=np.unique(distances)
    choices=rng.choice(levels,count-len(broad))
    pieces=[broad]
    for level in levels:
        n=int((choices==level).sum())
        if n: pieces.append(rng.choice(np.flatnonzero(distances==level),n,replace=True))
    selected=np.concatenate(pieces); rng.shuffle(selected)
    return pool[selected]


def materialize(corpus, rows):
    rows=np.asarray(rows,dtype=np.int64).reshape(-1,2)
    distances=corpus.distance(rows[:,0],rows[:,1])
    children=corpus.successors(rows[:,0]); valid=children>=0
    truth=np.full(children.shape,np.inf,dtype=np.float32)
    truth[valid]=corpus.distance(children[valid],np.broadcast_to(rows[:,1,None],children.shape)[valid])
    optimal=valid & np.isfinite(truth) & (truth+1==distances[:,None])
    return dict(tasks=rows,distances=distances,children=children,child_distances=truth,
                reachability=np.isfinite(distances).astype(np.int8),optimal=optimal)


def sample_goals(corpus,sources,rng,reserved,*,ranking=False,allowed=None,fallback=None,allow_missing=False,
                 local_fraction=0.,walk_max=None):
    """Fresh complete goals, using teacher labels only in the training sampler."""
    sources=np.asarray(sources,dtype=np.int64)
    if not 0<=local_fraction<=1 or (local_fraction and (hasattr(corpus,'scenes') or walk_max is None or walk_max<0)):
        raise ValueError('Local-goal proposals require a permutation graph and nonnegative walk bound')
    goals=np.empty(len(sources),dtype=np.int64); bad=np.ones(len(sources),dtype=bool)
    def forbidden(values):
        if not len(reserved): return np.zeros(values.shape,dtype=bool)
        at=np.searchsorted(reserved,values)
        return (at<len(reserved)) & (reserved[np.minimum(at,len(reserved)-1)]==values)
    scene_ids=corpus.locate(sources) if hasattr(corpus,'scenes') else None
    children=corpus.successors(sources) if ranking else None
    for _ in range(64):
        if not bad.any(): break
        if scene_ids is None:
            goals[bad]=rng.choice(allowed,int(bad.sum()))
            if local_fraction:
                local=np.flatnonzero(bad & (rng.random(len(sources))<local_fraction))
                goals[local]=sources[local]
                lengths=rng.integers(0,walk_max+1,len(local))
                for step in range(int(lengths.max(initial=0))):
                    ids=local[lengths>step]
                    goals[ids]=corpus.children[goals[ids],rng.integers(corpus.max_candidates,size=len(ids))]
        else:
            for sid in np.unique(scene_ids[bad]):
                mask=bad & (scene_ids==sid); scene=corpus.scenes[sid]
                pool=scene.goals[:scene.metadata['training_goals']]+corpus.offsets[sid]
                goals[mask]=rng.choice(pool,int(mask.sum()))
        bad=forbidden(keys(sources,goals,corpus.size))
        if local_fraction:
            at=np.searchsorted(allowed,goals)
            bad|=(at==len(allowed)) | (allowed[np.minimum(at,len(allowed)-1)]!=goals)
        if ranking:
            labels=corpus.distance(sources,goals)
            bad|=~np.isfinite(labels) | (labels==0)
            legal=children>=0
            relations=np.full(children.shape,-1,dtype=np.int64)
            relations[legal]=keys(children[legal],np.broadcast_to(goals[:,None],children.shape)[legal],corpus.size)
            bad|=forbidden(relations).any(axis=1)
    if bad.any() and fallback is not None:
        goals[bad]=fallback[bad]; bad[:]=False
    if bad.any() and not allow_missing: raise ValueError('No eligible non-leaking complete goal for training source')
    rows=np.column_stack([sources,goals])
    return (rows[~bad],~bad) if allow_missing else rows


def board_cohorts(corpus,config,rng):
    from grounded_llm.core.qmap.physical_data import group_seed
    spec=config['relations']
    names=('validation','test','unseen_goal_validation','unseen_goal_test',
           'unseen_layout_validation','unseen_layout_test','original_layout_test',
           'original_layout_validation','diagnostic','diagnostic_original')
    groups={name:[] for name in names}; reserved=[]; train_specs=[]
    for sid,scene in enumerate(corpus.scenes):
        offset=corpus.offsets[sid]; split=scene.metadata['split']; nt=scene.metadata['training_goals']
        blocked=scene.goals[nt:]
        safe_parent=~np.isin(np.arange(len(scene.states)),blocked)
        safe_rank=safe_parent & ~np.isin(scene.children,blocked).any(axis=1)
        source_key=json.dumps(scene.metadata['source'],sort_keys=True)
        eval_columns=np.arange(min(nt,spec.get('evaluation_training_goals',nt)))
        if split=='train':
            local_reserved=[]
            for cohort in ('validation','test'):
                rows=[]
                local_rng=np.random.default_rng(group_seed(source_key,config['seed'],'queries:'+cohort))
                for column in eval_columns:
                    g=scene.goals[column]
                    pool=np.flatnonzero(np.isfinite(scene.distances[column]) & (scene.distances[column]>0) & safe_rank)
                    proposed=np.column_stack([pool+offset,np.full(len(pool),g+offset,dtype=np.int64)])
                    if local_reserved:
                        held=np.concatenate(local_reserved); kids=corpus.successors(proposed[:,0]); legal=kids>=0
                        candidate_keys=np.full(kids.shape,-1,dtype=np.int64)
                        candidate_keys[legal]=keys(kids[legal],np.broadcast_to(proposed[:,1,None],kids.shape)[legal],corpus.size)
                        bad=np.isin(keys(proposed[:,0],proposed[:,1],corpus.size),held)
                        bad|=np.isin(candidate_keys,held).any(axis=1); proposed=proposed[~bad]
                    if not len(proposed): continue
                    rows.append(balanced(proposed,spec['holdout_per_goal'],corpus.distance(proposed[:,0],proposed[:,1]),local_rng))
                if rows:
                    rows=np.concatenate(rows); groups[cohort].append(rows)
                    local_reserved.append(closure(corpus,rows)); reserved.append(local_reserved[-1])
            heldout=len(scene.goals)-nt
            for column in range(nt,len(scene.goals)):
                cohort='unseen_goal_validation' if column-nt<(heldout+1)//2 else 'unseen_goal_test'
                local_rng=np.random.default_rng(group_seed(source_key,config['seed'],'queries:'+cohort+str(column)))
                g=scene.goals[column]; finite=np.isfinite(scene.distances[column]); pool=np.arange(len(scene.states))
                rows=np.column_stack([pool+offset,np.full(len(pool),g+offset,dtype=np.int64)])
                selected=balanced(rows[finite],spec['evaluation_per_goal'],scene.distances[column,finite],local_rng)
                if (~finite).any():
                    bad=local_rng.choice(np.flatnonzero(~finite),spec['unreachable_eval_per_goal'],replace=True)
                    selected=np.concatenate([selected,rows[bad]])
                groups[cohort].append(selected)
            train_specs.append((sid,safe_parent,safe_rank))
        else:
            cohort='diagnostic' if split=='diagnostic' else 'unseen_layout_'+split
            columns=np.linspace(0,len(scene.goals)-1,min(len(scene.goals),spec.get('evaluation_training_goals',8)),dtype=int)
            for column in columns:
                local_rng=np.random.default_rng(group_seed(source_key,config['seed'],'queries:'+cohort+str(column)))
                g=scene.goals[column]; finite=np.isfinite(scene.distances[column]); pool=np.flatnonzero(finite)
                rows=np.column_stack([pool+offset,np.full(len(pool),g+offset,dtype=np.int64)])
                selected=balanced(rows,spec['evaluation_per_goal'],scene.distances[column,pool],local_rng)
                if (~finite).any():
                    bad=local_rng.choice(np.flatnonzero(~finite),spec['unreachable_eval_per_goal'],replace=True)
                    selected=np.concatenate([selected,np.column_stack([bad+offset,np.full(len(bad),g+offset)])])
                groups[cohort].append(selected)
            original=int(scene.metadata['original_state'])+offset
            if split=='diagnostic':
                groups['diagnostic_original'].append(np.array([[original,int(scene.metadata['diagnostic_goal'])+offset]]))
            else:
                groups['original_layout_'+split].append(np.column_stack([np.full(len(scene.goals),original),scene.goals+offset]))
    reserved=np.unique(np.concatenate(reserved))
    pairs,decisions=[],[]
    for sid,safe_parent,safe_rank in train_specs:
        scene=corpus.scenes[sid]; offset=corpus.offsets[sid]; nt=scene.metadata['training_goals']
        parents=np.flatnonzero(safe_parent)+offset
        rows,_=sample_goals(corpus,parents,rng,reserved,allow_missing=True)
        pairs.append(rows)
        reachable=np.isfinite(scene.distances[:nt]).any(axis=0)
        positive=(scene.distances[:nt]>0).any(axis=0)
        parents=np.flatnonzero(safe_rank & reachable & positive)+offset
        rows,_=sample_goals(corpus,parents,rng,reserved,ranking=True,allow_missing=True)
        decisions.append(rows)
    return {k:np.concatenate(v) for k,v in groups.items() if v},np.concatenate(pairs),np.concatenate(decisions),reserved,None


def token_cohorts(corpus,config,rng):
    spec=config['relations']; partition=np.zeros(corpus.size,dtype=np.int8)
    order=rng.permutation(corpus.size)
    held=round(corpus.size*spec['endpoint_holdout_fraction'])
    partition[order[:held]]=1; partition[order[held:2*held]]=2
    pools=[np.flatnonzero(partition==i) for i in range(3)]
    safe=np.flatnonzero((partition==0) & (partition[corpus.children]==0).all(axis=1))
    def proposals(count,source_pool,goal_pool,local=False):
        g=rng.choice(goal_pool,count); s=rng.choice(source_pool,count)
        if local:
            half=count//2; s[:half]=g[:half]
            lengths=rng.integers(0,2*corpus.metadata['statistics']['diameter']+1,half)
            for step in range(int(lengths.max())):
                ids=np.flatnonzero(lengths>step)
                s[ids]=corpus.children[s[ids],rng.integers(corpus.max_candidates,size=len(ids))]
        return np.column_stack([s,g])
    groups={}; reserved=[]
    for name in ('validation','test'):
        rows=proposals(spec['evaluation_count']*4,safe,pools[0])
        if reserved:
            children=corpus.successors(rows[:,0])
            bad=np.isin(keys(rows[:,0],rows[:,1],corpus.size),reserved[0])
            bad|=np.isin(keys(children,rows[:,1,None],corpus.size),reserved[0]).any(axis=1)
            rows=rows[~bad]
        groups[name]=balanced(rows,spec['evaluation_count'],corpus.distance(rows[:,0],rows[:,1]),rng)
        reserved.append(closure(corpus,groups[name]))
    for suffix,pool in [('validation',pools[1]),('test',pools[2])]:
        groups['unseen_endpoint_'+suffix]=proposals(spec['evaluation_count'],pool,pools[0])
        groups['unseen_both_'+suffix]=proposals(spec['evaluation_count'],pool,pool)
    suite=config.get('data',{}).get('diagnostic_suite')
    if suite:
        from grounded_llm.tasks.token_swapping.relation_data import rank
        cases=json.loads(Path(suite).read_text(encoding='utf-8'))['cases']
        rows=[]
        for case in cases:
            if case['task']=='token_swapping':
                rows.append([int(rank(np.asarray(case['initial']))),int(rank(np.asarray(case['goal'])))])
        groups['diagnostic']=np.asarray(rows,dtype=np.int64)
        reserved.append(closure(corpus,groups['diagnostic']))
    reserved=np.unique(np.concatenate(reserved))
    if spec.get('complete_training_sources'):
        pairs=sample_goals(corpus,pools[0],rng,reserved,allowed=pools[0])
        decisions=sample_goals(corpus,safe,rng,reserved,ranking=True,allowed=pools[0])
        return groups,pairs,decisions,reserved,partition
    rows=proposals(spec['point_count']*5,pools[0],pools[0],local=True)
    rows=rows[(partition[rows[:,0]]==0) & ~np.isin(keys(rows[:,0],rows[:,1],corpus.size),reserved)]
    pairs=balanced(rows,spec['point_count'],corpus.distance(rows[:,0],rows[:,1]),rng)
    rows=proposals(spec['decision_count']*8,safe,pools[0],local=True)
    rows=rows[np.isin(rows[:,0],safe)]
    children=corpus.successors(rows[:,0])
    bad=np.isin(keys(rows[:,0],rows[:,1],corpus.size),reserved)
    bad|=np.isin(keys(children,rows[:,1,None],corpus.size),reserved).any(axis=1)
    rows=rows[~bad & (rows[:,0]!=rows[:,1])]
    decisions=balanced(rows,spec['decision_count'],corpus.distance(rows[:,0],rows[:,1]),rng)
    return groups,pairs,decisions,reserved,partition


def prepare(config,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    module=importlib.import_module(f'grounded_llm.tasks.{config["task"]}.relation_data')
    corpus=module.load(out/'graphs') if (out/'graphs'/'corpus.json').exists() or (out/'graphs'/'oracle.json').exists() else module.build(config,out/'graphs')
    rng=np.random.default_rng(config['seed']+1)
    groups,pairs,decisions,reserved,partition=(token_cohorts if config['task']=='token_swapping' else board_cohorts)(corpus,config,rng)
    data={'pairs_train':pairs,'pair_distances_train':corpus.distance(pairs[:,0],pairs[:,1]),'reserved_relations':reserved}
    for name,rows in [('train',decisions),*groups.items()]:
        for key,value in materialize(corpus,rows).items(): data[f'{key}_{name}']=value
    finite=np.concatenate([v[np.isfinite(v)] for k,v in data.items() if k.startswith('distances_') or k.startswith('child_distances_') or k=='pair_distances_train'])
    maximum=max(float(finite.max()),float(corpus.metadata['statistics']['diameter']) if config['task']=='token_swapping'
                else max(scene.metadata['finite_max'] for scene in corpus.scenes))
    if maximum>=config['training']['unreachable_target'] or maximum>=config['model']['cap']:
        raise ValueError(('Finite oracle costs exceed the declared encoding/proxy scale',maximum))
    if np.isin(keys(pairs[:,0],pairs[:,1],corpus.size),reserved).any() or np.intersect1d(closure(corpus,decisions),reserved).size:
        raise ValueError('Held-out relation closure leaked into training')
    if partition is not None:
        np.save(out/'endpoint_partition.npy',partition)
        exposed=np.concatenate([pairs.ravel(),decisions[:,1],data['children_train'].ravel()])
        if np.any(partition[exposed[exposed>=0]]!=0): raise ValueError('Held-out endpoint used by either loss')
    np.savez_compressed(out/'relations.npz',**data)
    summary=dict(task=config['task'],states=corpus.size,input_dim=corpus.input_dim,max_candidates=corpus.max_candidates,
                 seed=config['seed'],finite_max=maximum,unknown_training_labels=0,
                 pair_count=len(pairs),decision_count=len(decisions),reserved_relations=len(reserved),
                 cohorts={name:dict(count=len(rows),reachable=int(np.isfinite(data[f'distances_{name}']).sum())) for name,rows in groups.items()},
                 selected_train_states=len(np.unique(pairs[:,0])),
                 all_selected_sources_in_training=bool(config['relations'].get('complete_training_sources',False)))
    atomic_json(out/'summary.json',summary)
    return corpus,data,summary


def load(config,out):
    out=Path(out)
    corpus=importlib.import_module(f'grounded_llm.tasks.{config["task"]}.relation_data').load(out/'graphs')
    data=dict(np.load(out/'relations.npz'))
    return corpus,data,json.loads((out/'summary.json').read_text())
