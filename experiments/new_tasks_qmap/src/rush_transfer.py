"""Frozen cross-layout comparison of rule-derived two-action blocking contexts."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.qmap.sampling import CoverageSampler
from .relation_data import load,training_decision_mask,materialize,FailureReplaySampler
from .relation_diagnose import training_start_queries
from .relation_evaluate import QCache, planning_ids, safe_scores


def blocking(env,state,goal):
    masks=[env.masks[i][int(p)] for i,p in enumerate(state)]
    corridor=[]
    for i,(a,b) in enumerate(zip(state,goal)):
        value=0
        for p in range(min(int(a),int(b)),max(int(a),int(b))+1):value|=env.masks[i][p]
        corridor.append(value)
    return np.array([[i!=j and bool(corridor[i]&masks[j]) for j in range(len(state))]
                     for i in range(len(state))],dtype=bool)


def action_context(env,state,goal,children):
    state=np.asarray(state,dtype=int);goal=np.asarray(goal,dtype=int)
    children=np.asarray(children,dtype=int)
    changed=children!=state
    if not (changed.sum(-1)==1).all():raise ValueError('Each Rush Hour action moves exactly one vehicle')
    movers=changed.argmax(-1);displacements=children[np.arange(len(children)),movers]-state[movers]
    low=np.zeros(len(state),dtype=int);high=low.copy()
    for mover,delta in zip(movers,displacements):low[mover]=min(low[mover],delta);high[mover]=max(high[mover],delta)
    before=blocking(env,state,goal);after=[blocking(env,c,goal) for c in children]
    return state,goal,movers,displacements,low,high,before,after


def context_key(env,context,a,b,level):
    state,goal,movers,delta,low,high,before,after=context
    i,j=int(movers[a]),int(movers[b]);n=len(state)
    if level=='direct':
        def role(k,action):
            label,horizontal,length,_=env.pieces[k]
            difference=int(goal[k]-state[k]);step=int(delta[action])
            return (int(label=='A'),int(horizontal),length,int(np.sign(difference)),
                    int(np.sign(step)),min(abs(step),2),int(np.sign(difference)*step>0),
                    int(before[k].sum()),int(before[:,k].sum()),
                    int(after[action][k].sum()),int(after[action][:,k].sum()))
        return (n,int(i==j),role(i,a),role(j,b),
                tuple(int(matrix[x,y]) for matrix in (before,after[a],after[b]) for x,y in ((i,j),(j,i))))
    if level!='neighborhood':raise ValueError('Unknown blocking context level')
    matrices=(before,after[a],after[b]);support={i,j}
    for matrix in matrices:
        for k in (i,j):support.update(np.flatnonzero(matrix[k]|matrix[:,k]).tolist())
    def position(k):
        _,horizontal,_,lane=env.pieces[k]
        return (int(state[k]),lane) if horizontal else (lane,int(state[k]))
    origin=position(i);descriptions=[]
    for k in support:
        label,horizontal,length,_=env.pieces[k];x,y=position(k)
        outside=[q for q in range(n) if q not in support]
        node=(int(k==i),int(k==j),int(label=='A'),int(horizontal),length,x-origin[0],y-origin[1],
              int(goal[k]-state[k]),int(low[k]),int(high[k]),
              tuple(int(matrix[k,outside].sum()) for matrix in matrices),
              tuple(int(matrix[outside,k].sum()) for matrix in matrices))
        descriptions.append((node,k))
    descriptions.sort();order=[k for _,k in descriptions]
    edges=tuple(tuple(int(matrix[x,y]) for x in order for y in order) for matrix in matrices)
    return (n,int(delta[a]),int(delta[b]),tuple(node for node,_ in descriptions),edges)


def canonical_pair(env,context,a,b,level):
    left=context_key(env,context,a,b,level);right=context_key(env,context,b,a,level)
    return (left,1) if left<=right else (right,-1)


def add_observation(bank,key,truth,prediction,scene,example):
    row=bank.get(key)
    if row is None:
        row=dict(count=np.zeros(3,dtype=np.int64),correct=np.zeros(3,dtype=np.int64),scenes=set(),examples={},
                 layout_count=defaultdict(lambda:np.zeros(3,dtype=np.int64)),
                 layout_correct=defaultdict(lambda:np.zeros(3,dtype=np.int64)),
                 layout_examples={},layout_errors={})
        bank[key]=row
    slot=int(truth)+1;row['count'][slot]+=1;row['correct'][slot]+=int(prediction==truth)
    row['scenes'].add(scene);row['examples'].setdefault(int(truth),example)
    row['layout_count'][scene][slot]+=1;row['layout_correct'][scene][slot]+=int(prediction==truth)
    if prediction==truth:row['examples'][int(truth)]=example
    row['layout_examples'].setdefault((scene,int(truth)),example)
    if prediction!=truth:row['layout_errors'].setdefault((scene,int(truth)),example)


def summarize(train,held,minimum_training_layouts=2):
    result=dict(non_tie_held_pairs=len(held),matched_pairs=0,matched_across_multiple_training_layouts=0,
                training_label_varies=0,held_label_differs_from_consistent_training=0,
                same_label=0,train_fitted_pairs=0,train_fitted_held_correct=0,
                train_underfit_pairs=0,train_underfit_held_correct=0)
    matched_queries=set();stable_queries=set();fitted_queries=set();examples=[];ambiguous=[]
    for key,truth,prediction,query,example in held:
        row=train.get(key)
        if row is None:continue
        result['matched_pairs']+=1;matched_queries.add(query)
        if len(row['scenes'])<minimum_training_layouts:continue
        result['matched_across_multiple_training_layouts']+=1
        labels=np.flatnonzero(row['count'])-1
        if len(labels)!=1:
            result['training_label_varies']+=1
            if len(ambiguous)<3:ambiguous.append(dict(training=[row['examples'][int(label)] for label in labels],held=example))
            continue
        if labels[0]!=truth:
            result['held_label_differs_from_consistent_training']+=1;continue
        result['same_label']+=1;stable_queries.add(query);slot=truth+1
        accuracy=min(float(row['layout_correct'][scene][slot]/row['layout_count'][scene][slot])
                     for scene in row['scenes'])
        if accuracy>=.9:
            result['train_fitted_pairs']+=1;result['train_fitted_held_correct']+=int(prediction==truth)
            fitted_queries.add(query)
            if prediction!=truth and len(examples)<5:
                examples.append(dict(training=row['examples'][truth],held=example,
                    training_pair_accuracy=accuracy,training_layouts=len(row['scenes'])))
        else:
            result['train_underfit_pairs']+=1;result['train_underfit_held_correct']+=int(prediction==truth)
    result.update(matched_queries=len(matched_queries),same_label_queries=len(stable_queries),
                  train_fitted_held_queries=len(fitted_queries),examples=examples,ambiguous_examples=ambiguous,
                  train_fitted_held_accuracy=(result['train_fitted_held_correct']/result['train_fitted_pairs'] if result['train_fitted_pairs'] else None),
                  train_underfit_held_accuracy=(result['train_underfit_held_correct']/result['train_underfit_pairs'] if result['train_underfit_pairs'] else None))
    return result


def training_groups(banks):
    """Full training queries paired by local context; held observations are absent."""
    result=[]
    for level,bank in banks.items():
        for row in bank.values():
            if len(row['scenes'])<2:continue
            entries=list(row['layout_examples'].items());labels=np.flatnonzero(row['count'])-1
            if len(labels)>1:
                pair=next(((a,b) for a in entries for b in entries
                           if a[0][0]!=b[0][0] and a[0][1]!=b[0][1]),None)
                if pair:result.append(('global_context_contrast',level,pair[0][1],pair[1][1]))
            elif labels[0]!=0:
                label=int(labels[0]);slot=label+1
                accuracy=min(row['layout_correct'][scene][slot]/row['layout_count'][scene][slot] for scene in row['scenes'])
                if accuracy>=.9:
                    a=entries[0];b=next(entry for entry in entries if entry[0][0]!=a[0][0])
                    result.append(('retain_consistent_relation',level,a[1],b[1]))
                elif row['layout_errors']:
                    identity,a=next(iter(row['layout_errors'].items()))
                    b=next(entry for entry in entries if entry[0][0]!=identity[0])
                    result.append(('training_rank_error',level,a,b[1]))
    return result


def export_training_groups(corpus,data,banks,path):
    groups=training_groups(banks)
    if not groups:return dict(groups=0,queries=0)
    tasks=np.unique(np.array([[case['source'],case['goal']] for _,_,a,b in groups for case in (a,b)],dtype=np.int64),axis=0)
    if not training_decision_mask(corpus,data,tasks).all():raise ValueError('Structural groups cross a held relation')
    lookup={tuple(row):i for i,row in enumerate(tasks.tolist())}
    paired=np.array([[lookup[(case['source'],case['goal'])] for case in (a,b)] for _,_,a,b in groups])
    kinds=[kind for kind,_,_,_ in groups]
    packed=materialize(corpus,tasks)
    np.savez_compressed(path,**packed,group_tasks=paired,group_kind=np.array(kinds),
        group_level=np.array([level for _,level,_,_ in groups]),
        group_action_slots=np.array([[a['slots'],b['slots']] for _,_,a,b in groups]),
        group_comparison_direction=np.array([[a['canonical_direction'],b['canonical_direction']] for _,_,a,b in groups]))
    return dict(groups=len(groups),queries=len(tasks),groups_by_kind={kind:kinds.count(kind) for kind in sorted(set(kinds))},
                all_queries_training_allowed=True,all_legal_candidates_preserved=True,
                grouping_uses_training_queries_only=True,optimizer_updates=0)


class StructuralReplaySampler(FailureReplaySampler):
    """Keep both complete queries of a cross-layout comparison in one batch."""
    def __init__(self,corpus,data,bank,groups_per_batch,seed,state=None):
        self.corpus=corpus;self.bank=bank;self.groups_per_batch=groups_per_batch
        self.tasks=np.asarray(bank['tasks']);self.children=bank['children'];self.groups=bank['group_tasks']
        if groups_per_batch<1 or not len(self.groups) or self.groups.shape[1:]!=(2,):raise ValueError('Nonempty two-query groups required')
        if not training_decision_mask(corpus,data,self.tasks).all():raise ValueError('Structural groups cross a held relation')
        exact=materialize(corpus,self.tasks)
        for key,value in exact.items():
            if not np.array_equal(bank[key],value):raise ValueError(('Structural labels or physical candidates changed',key))
        if (self.groups<0).any() or (self.groups>=len(self.tasks)).any():raise ValueError('Invalid structural query index')
        self.rng=np.random.default_rng(seed)
        self.coverage=CoverageSampler(np.zeros(len(self.groups),dtype=np.uint8),self.rng,len(self.groups),1.)
        self.draws=np.zeros(len(self.groups),dtype=np.uint32);self.last_annotations={}
        if state is not None:
            self.coverage.load_state_dict(state['coverage']);self.draws=state['draws'].copy()
            self.rng.bit_generator.state=deepcopy(state['rng'])

    def apply(self,pairs,point_indices,tasks,rank_indices,point_index,rank_index,point_origin,rank_origin,finite_fraction):
        groups=self.coverage.draw(self.groups_per_batch);ids=self.groups[groups].ravel();n=len(ids)
        retained=np.flatnonzero(rank_origin!=0);limit=int(retained[0]) if len(retained) else len(tasks)
        if n>limit:raise ValueError('Structural groups exceed the remaining non-path quota')
        count=n+int((self.children[ids]>=0).sum())
        available=np.arange(len(pairs)-round(len(pairs)*finite_fraction),len(pairs))
        available=available[point_origin[available]==0]
        truth=self.corpus.distance(pairs[available,0],pairs[available,1])
        positions=np.r_[self.rng.permutation(available[truth>5]),self.rng.permutation(available[(truth>2)&(truth<=5)]),
                        self.rng.permutation(available[truth<=2])][:count]
        if len(positions)!=count:raise ValueError('Structural groups exceed the finite point quota')
        p,pi,t,ti,po,ro,_=self.replace(pairs,point_indices,tasks[:limit],rank_indices[:limit],point_index,rank_index,
                                    ids,np.ones(n,dtype=np.uint8),point_positions=positions)
        tasks=tasks.copy();rank_indices=rank_indices.copy();tasks[:limit]=t;rank_indices[:limit]=ti
        point_origin=point_origin.copy();rank_origin=rank_origin.copy();point_origin[po>0]=7;rank_origin[:limit][ro>0]=7
        self.last_annotations={key:np.full(len(tasks),-1,dtype=np.int64) for key in ('rank_structure_group','rank_structure_member')}
        self.last_annotations['rank_structure_group'][limit-n:limit]=np.repeat(groups,2)
        self.last_annotations['rank_structure_member'][limit-n:limit]=np.tile([0,1],len(groups))
        np.add.at(self.draws,groups,1)
        return p,pi,tasks,rank_indices,point_origin,rank_origin

    def state_dict(self):
        return dict(bank=self.bank,coverage=self.coverage.state_dict(),draws=self.draws.copy(),rng=deepcopy(self.rng.bit_generator.state))

    def report(self):
        return dict(groups=len(self.groups),queries=len(self.tasks),contexts_per_batch=2*self.groups_per_batch,
                    group_draws=int(self.draws.sum()),minimum_group_draws=int(self.draws.min()),
                    unique_groups_drawn=int((self.draws>0).sum()),passes=self.coverage.passes)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--training-queries',type=int,default=16384);parser.add_argument('--pairs-per-training-query',type=int,default=12)
    parser.add_argument('--training-trace',type=Path,nargs='+')
    parser.add_argument('--additional-training-bank',type=Path,nargs='+',default=[])
    parser.add_argument('--training-root-queries',type=int,default=0)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True);torch.set_num_threads(2);started=time.time()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=saved['config']
    corpus,data,_=load(config,args.data)
    model=StateEncoder(input_dim=saved['input_dim'],**config['model']).to(args.device)
    model.load_state_dict(saved['model']);model.eval().requires_grad_(False);cache=QCache(model,corpus,args.device)
    if args.training_trace:
        pieces=[];trace_rng=np.random.default_rng(config['seed']+91003);trace_sources=[]
        for i,path in enumerate(args.training_trace):
            with np.load(path) as packed:
                tasks=packed['tasks'];steps=packed['rank_step'];eligible=np.flatnonzero(steps<=saved['step'])
                quota=args.training_queries//len(args.training_trace)+(i<args.training_queries%len(args.training_trace))
                chosen=trace_rng.choice(eligible,min(quota,len(eligible)),replace=False)
                pieces.append(tasks[chosen]);trace_sources.append(dict(path=str(path),minimum_step=int(steps[chosen].min()),maximum_step=int(steps[chosen].max()),sampled=len(chosen)))
        training=np.unique(np.concatenate(pieces),axis=0)
        safe=training_decision_mask(corpus,data,training)
        if not safe.all():raise ValueError('Recorded training task crosses a protected boundary')
    else:
        rows,truth=training_start_queries(corpus,data,'original_starts')
        selected=planning_ids(corpus,rows,truth,args.training_queries,np.random.default_rng(config['seed']+91003),[1,2,5,10,20,40,80,256])
        training=rows[selected];trace_sources=[]
    extra_sources=[];extra=[training]
    for path in args.additional_training_bank:
        with np.load(path) as archive:rows=archive['tasks']
        if not training_decision_mask(corpus,data,rows).all():raise ValueError('Additional training bank crosses a protected boundary')
        extra.append(rows);extra_sources.append(dict(path=str(path),queries=len(rows)))
    if args.training_root_queries:
        rows,truth=training_start_queries(corpus,data,'original_starts')
        selected=planning_ids(corpus,rows,truth,args.training_root_queries,np.random.default_rng(config['seed']+91009),[1,2,5,10,20,40,80,256])
        extra.append(rows[selected]);extra_sources.append(dict(source='permitted_original_training_roots',queries=len(selected)))
    training=np.unique(np.concatenate(extra),axis=0)
    name='original_layout_validation';pool=data['tasks_'+name]
    held=pool[planning_ids(corpus,pool,data['distances_'+name],config['evaluation']['rollout_count'],np.random.default_rng(config['seed']+len(name)))]
    np.savez_compressed(args.out/'queries.npz',training=training,held=held)
    levels=('direct','neighborhood');banks={level:{} for level in levels};observations={level:[] for level in levels}
    rng=np.random.default_rng(config['seed']+91007);totals={}
    for fold,queries in [('training',training),('held',held)]:
        evaluated=0;parents=0
        for begin in range(0,len(queries),256):
            batch=queries[begin:begin+256];children=corpus.successors(batch[:,0]);scores=safe_scores(cache,children,batch[:,1]);legal=children>=0
            targets=np.broadcast_to(batch[:,1,None],children.shape)
            truth=np.full(children.shape,np.inf,dtype=np.float32);truth[legal]=corpus.distance(children[legal],targets[legal])
            scene_ids=corpus.locate(batch[:,0]);parent_truth=corpus.distance(batch[:,0],batch[:,1])
            for k,(source,goal) in enumerate(batch):
                sid=int(scene_ids[k]);scene=corpus.scenes[sid];offset=corpus.offsets[sid];slots=np.flatnonzero(legal[k])
                catalogue=scene.metadata.get('catalogue_group',str(sid))
                state=scene.states[source-offset];target=scene.states[goal-offset];physical=scene.states[children[k,slots]-offset]
                context=action_context(scene.env,state,target,physical)
                pairs=[(a,b) for a in range(len(slots)) for b in range(a+1,len(slots))]
                if fold=='training' and len(pairs)>args.pairs_per_training_query:
                    pairs=[pairs[i] for i in rng.choice(len(pairs),args.pairs_per_training_query,replace=False)]
                parents+=1
                for a,b in pairs:
                    va,vb=slots[a],slots[b];label=int(np.sign(truth[k,va]-truth[k,vb]))
                    prediction=int(np.sign(scores[k,va]-scores[k,vb]))
                    if fold=='held' and label==0:continue
                    optimal=(truth[k]+1==parent_truth[k]);primary=bool(optimal[va]!=optimal[vb])
                    example=dict(scene=sid,source=int(source),goal=int(goal),actions=[corpus.action(int(source),int(va)),corpus.action(int(source),int(vb))],
                        slots=[int(va),int(vb)],truth=[float(truth[k,va]),float(truth[k,vb])],scores=[float(scores[k,va]),float(scores[k,vb])],optimal_vs_other=primary)
                    for level in levels:
                        key,direction=canonical_pair(scene.env,context,a,b,level)
                        presented=dict(example,canonical_direction=direction)
                        if fold=='training':add_observation(banks[level],key,label*direction,prediction*direction,catalogue,presented)
                        else:observations[level].append((key,label*direction,prediction*direction,begin+k,presented))
                    evaluated+=1
            if begin%2048==0:print(json.dumps(dict(fold=fold,queries=begin+len(batch),pairs=evaluated,seconds=time.time()-started)),flush=True)
        scenes=np.unique(corpus.locate(queries[:,0]))
        totals[fold]=dict(queries=parents,pairs=evaluated,scenes=len(scenes),
                         catalogue_groups=len({corpus.scenes[int(sid)].metadata.get('catalogue_group',str(sid)) for sid in scenes}))
    result=dict(step=saved['step'],source_commit=saved['source_commit'],checkpoint=str(args.checkpoint),sampling=totals,
        training_trace_sources=trace_sources,additional_training_sources=extra_sources,levels={},seconds=time.time()-started,
        condition='Frozen original936input128D Q. Rule-derived local contexts are diagnostic grouping only. Every candidate remains available. Training labels include ties; held strict-order comparisons require matched training contexts from at least two layouts. Local context identity is not whole-puzzle equivalence. No optimizer update or validation replay.')
    for level in levels:
        result['levels'][level]=dict(all_strict_pairs=summarize(banks[level],observations[level]),
            optimal_vs_other=summarize(banks[level],[row for row in observations[level] if row[-1]['optimal_vs_other']]),training_contexts=len(banks[level]))
    for groups in result['levels'].values():
        for subset in ('all_strict_pairs','optimal_vs_other'):
            report=groups[subset]
            cases=[case for example in report['examples'] for case in (example['training'],example['held'])]
            cases.extend(case for example in report['ambiguous_examples'] for case in example['training']+[example['held']])
            for case in cases:
                sid=case['scene'];scene=corpus.scenes[sid];offset=corpus.offsets[sid]
                case['source_board']=scene.env.render(tuple(map(int,scene.states[case['source']-offset])))
                case['goal_board']=scene.env.render(tuple(map(int,scene.states[case['goal']-offset])))
    result['prepared_training_groups']=export_training_groups(corpus,data,banks,args.out/'training_structure_groups.npz')
    atomic_json(args.out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
