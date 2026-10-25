"""Prepare exact state-pair cohorts while excluding held-out relation closures."""
import importlib
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.sampling import CoverageSampler
from grounded_llm.core.qmap.exact_graph import edge_costs


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
    optimal=valid & np.isfinite(truth) & (truth+edge_costs(corpus,rows[:,0])==distances[:,None])
    return dict(tasks=rows,distances=distances,children=children,child_distances=truth,
                reachability=np.isfinite(distances).astype(np.int8),optimal=optimal)


def training_decision_mask(corpus,data,rows):
    """Permit board decisions only inside the original training information set.

    Mining may visit sources outside the prepared source bank. The original
    layout, complete-goal and held-out relation-closure boundaries still apply.
    This checks entire decisions; it never removes individual legal actions.
    """
    rows=np.asarray(rows,dtype=np.int64).reshape(-1,2)
    if not hasattr(corpus,'scenes'): raise ValueError('Board corpus required')
    result=np.zeros(len(rows),dtype=bool)
    if not len(rows): return result
    if ((rows<0)|(rows>=corpus.size)).any(): raise ValueError('State outside corpus')
    source_scene=corpus.locate(rows[:,0]); goal_scene=corpus.locate(rows[:,1])
    children=corpus.successors(rows[:,0]); legal=children>=0
    reserved=data['reserved_relations']
    def forbidden(values):
        if not len(reserved): return np.zeros(values.shape,dtype=bool)
        at=np.searchsorted(reserved,values)
        return (at<len(reserved)) & (reserved[np.minimum(at,len(reserved)-1)]==values)
    blocked=forbidden(keys(rows[:,0],rows[:,1],corpus.size))
    related=np.full(children.shape,-1,dtype=np.int64)
    related[legal]=keys(children[legal],np.broadcast_to(rows[:,1,None],children.shape)[legal],corpus.size)
    blocked|=forbidden(related).any(axis=1)
    for sid in np.unique(source_scene):
        scene=corpus.scenes[int(sid)]
        if scene.metadata['split']!='train': continue
        ids=np.flatnonzero((source_scene==sid)&(goal_scene==sid)); offset=corpus.offsets[sid]
        nt=scene.metadata['training_goals']; heldout=scene.goals[nt:]+offset
        endpoint_safe=(~np.isin(rows[ids,0],heldout) & ~np.isin(children[ids],heldout).any(axis=1)
                       if bool(data.get('holdout_goal_endpoints',True)) else np.ones(len(ids),dtype=bool))
        allowed=(np.isin(rows[ids,1],scene.goals[:nt]+offset) & endpoint_safe
                 & ~blocked[ids] & legal[ids].any(axis=1))
        truth=corpus.distance(rows[ids,0],rows[ids,1])
        result[ids]=allowed & np.isfinite(truth) & (truth>0)
    return result


def sample_goals(corpus,sources,rng,reserved,*,ranking=False,allowed=None,fallback=None,allow_missing=False,
                 local_fraction=0.,walk_max=None,distance_bins=None,proposal='walk'):
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
    desired=None
    if distance_bins is not None:
        if scene_ids is not None:raise ValueError('Distance-layer proposals require a permutation oracle')
        distance_bins=np.asarray(distance_bins,dtype=np.float32)
        if np.any(np.diff(distance_bins)<=0):raise ValueError('Increasing exact distance bins required')
        desired=rng.integers(len(distance_bins),size=len(sources))
    if proposal not in ('walk','exact_layers') or (proposal=='exact_layers' and desired is None):
        raise ValueError('Exact-layer proposals require declared distance bins')
    for _ in range(64):
        if not bad.any(): break
        pending=np.flatnonzero(bad)
        if scene_ids is None:
            goals[bad]=(corpus.sample_layer_goals(sources[bad],distance_bins,desired[bad],rng)
                        if proposal=='exact_layers' else rng.choice(allowed,int(bad.sum())))
            if local_fraction and proposal=='walk':
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
        # Accepted goals never change. Recheck only resampled rows, preserving
        # proposal draws and RNG order while avoiding full-batch retries.
        rejected=forbidden(keys(sources[pending],goals[pending],corpus.size))
        if desired is not None:
            rejected|=np.searchsorted(distance_bins,corpus.distance(sources[pending],goals[pending]))!=desired[pending]
        if local_fraction or proposal=='exact_layers':
            at=np.searchsorted(allowed,goals[pending])
            rejected|=(at==len(allowed)) | (allowed[np.minimum(at,len(allowed)-1)]!=goals[pending])
        if ranking:
            labels=corpus.distance(sources[pending],goals[pending])
            rejected|=~np.isfinite(labels) | (labels==0)
            selected=children[pending];legal=selected>=0
            relations=np.full(selected.shape,-1,dtype=np.int64)
            relations[legal]=keys(selected[legal],np.broadcast_to(goals[pending,None],selected.shape)[legal],corpus.size)
            rejected|=forbidden(relations).any(axis=1)
        bad[:]=False;bad[pending]=rejected
    if bad.any() and fallback is not None:
        goals[bad]=fallback[bad]; bad[:]=False
    if bad.any() and not allow_missing: raise ValueError('No eligible non-leaking complete goal for training source')
    rows=np.column_stack([sources,goals])
    return (rows[~bad],~bad) if allow_missing else rows


class BoardRelationSampler:
    """Finite-distance strata and exact dead-boundary strata for one board run.

    Banks are deterministic functions of the existing training sources, goals
    and held-out closure. Only the caller's checkpointed RNG advances at draw.
    All physical point sources remain in the independent coverage traversal.
    """
    def __init__(self,corpus,data,options):
        if not hasattr(corpus,'scenes'): raise ValueError('Board relation sampling requires exact board scenes')
        self.corpus,self.data,self.options=corpus,data,options
        self.finite_fraction=options['finite_point_fraction']; self.boundary_fraction=options['boundary_rank_fraction']
        if not 0<self.finite_fraction<1 or not 0<=self.boundary_fraction<1: raise ValueError('Broad and finite quotas required; boundary quota may be zero on reversible graphs')
        self.point_index=np.full(corpus.size,-1,dtype=np.int32); self.rank_index=self.point_index.copy()
        self.point_index[data['pairs_train'][:,0]]=np.arange(len(data['pairs_train']))
        self.rank_index[data['tasks_train'][:,0]]=np.arange(len(data['tasks_train']))
        all_reserved=data['reserved_relations']
        def scene_bank(sid):
            scene=corpus.scenes[sid];offset=corpus.offsets[sid]
            start,stop=np.searchsorted(all_reserved,[offset*corpus.size,(offset+len(scene.states))*corpus.size])
            reserved=all_reserved[start:stop]
            def forbidden(values):
                if not len(reserved):return np.zeros(values.shape,dtype=bool)
                at=np.searchsorted(reserved,values)
                return (at<len(reserved)) & (reserved[np.minimum(at,len(reserved)-1)]==values)
            points=[];ranks=[];point_labels=[];rank_labels=[];boundaries=[]
            for j,g in enumerate(scene.goals[:scene.metadata['training_goals']]):
                source=np.flatnonzero(np.isfinite(scene.distances[j]))+offset; goal=np.full(len(source),g+offset)
                labels=scene.distances[j,source-offset]
                # Q(s) versus itself is identically zero under the distance
                # readout; do not spend the supplemental quota on zero-gradient
                # identity pairs. Incidental zeros remain in broad traversal.
                safe=(self.point_index[source]>=0) & (labels>0) & ~forbidden(keys(source,goal,corpus.size))
                points.append(np.column_stack([source[safe],goal[safe]])); point_labels.append(labels[safe])
                safe=(self.rank_index[source]>=0) & (labels>0) & ~forbidden(keys(source,goal,corpus.size))
                source,goal,labels=source[safe],goal[safe],labels[safe]
                child=corpus.successors(source); valid=child>=0
                related=np.full(child.shape,-1,dtype=np.int64)
                related[valid]=keys(child[valid],np.broadcast_to(goal[:,None],child.shape)[valid],corpus.size)
                safe=~forbidden(related).any(axis=1)
                child_truth=np.full(child.shape,np.inf,dtype=np.float32)
                child_truth[valid]=scene.distances[j,child[valid]-offset]
                boundary=(valid & np.isfinite(child_truth)).any(axis=1) & (valid & ~np.isfinite(child_truth)).any(axis=1)
                ranks.append(np.column_stack([source[safe],goal[safe]])); rank_labels.append(labels[safe]); boundaries.append(boundary[safe])
            return [np.concatenate(values) for values in (points,ranks,point_labels,rank_labels,boundaries)]
        ids=[sid for sid,scene in enumerate(corpus.scenes) if scene.metadata['split']=='train']
        workers=options.get('preparation_workers',1)
        if not 1<=workers<=32:raise ValueError('Preparation workers must be between1 and32')
        if workers>1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as executor:parts=list(executor.map(scene_bank,ids))
        else:parts=list(map(scene_bank,ids))
        points,ranks,point_labels,rank_labels,boundaries=map(list,zip(*parts))
        self.points=np.concatenate(points); self.ranks=np.concatenate(ranks)
        point_labels=np.concatenate(point_labels); rank_labels=np.concatenate(rank_labels); boundary=np.concatenate(boundaries)
        edges=np.asarray(options['distance_bin_upper_bounds'],dtype=np.float32)
        if np.any(np.diff(edges)<=0) or edges[-1]<max(point_labels.max(),rank_labels.max()): raise ValueError('Distance strata do not cover exact finite labels')
        def groups(labels,allowed,rows):
            bins=np.searchsorted(edges,labels)
            if options.get('balance_layouts'):
                scene_ids=corpus.locate(rows[:,0])
                result=[]
                # Banks are appended in corpus scene order; avoid rescanning
                # the complete bank once for every layout.
                for sid in np.unique(scene_ids[allowed]):
                    a,b=np.searchsorted(scene_ids,[sid,sid+1])
                    result.append([ids+a for j in range(len(edges))
                                   if len(ids:=np.flatnonzero(allowed[a:b] & (bins[a:b]==j)))])
                return result
            return [ids for j in range(len(edges)) if len(ids:=np.flatnonzero(allowed & (bins==j)))]
        def counts(labels,allowed):
            return {f'({-1 if j==0 else float(edges[j-1])},{float(upper)}]':int((allowed & (np.searchsorted(edges,labels)==j)).sum())
                    for j,upper in enumerate(edges)}
        self.point_groups=groups(point_labels,np.ones(len(point_labels),bool),self.points)
        self.boundary_groups=groups(rank_labels,boundary,self.ranks); self.other_groups=groups(rank_labels,~boundary,self.ranks)
        if (self.boundary_fraction and not self.boundary_groups) or not self.other_groups: raise ValueError('Requested exact boundary/nonboundary training relations unavailable')
        self.summary=dict(finite_point_bank=len(self.points),ranking_bank=len(self.ranks),boundary_ranking_bank=int(boundary.sum()),
            point_stratum_counts=counts(point_labels,np.ones(len(point_labels),bool)),boundary_stratum_counts=counts(rank_labels,boundary),
            nonboundary_stratum_counts=counts(rank_labels,~boundary),options=options,
            broad_source_fraction=1-self.finite_fraction,all_legal_candidates_preserved=True,
            finite_supplement_excludes_identity_pairs=True,
            holdout_closure_excluded=True,banks_use_only_original_training_source_rows=True)
        self.coverage_fraction=options.get('ranking_coverage_fraction',0.)
        if not 0<=self.coverage_fraction<1: raise ValueError('Ranking coverage must leave a stratified quota')
        self.coverage_sampler=None
        if self.coverage_fraction:
            self.coverage_rng=np.random.default_rng(options['coverage_seed'])
            self.coverage_sampler=CoverageSampler(np.zeros(len(self.ranks),dtype=np.uint8),self.coverage_rng,len(self.ranks),1.)
            self.summary['ranking_coverage_fraction']=self.coverage_fraction

    def coverage_state_dict(self):
        if self.coverage_sampler is None: return None
        return dict(sampler=self.coverage_sampler.state_dict(),rng=self.coverage_rng.bit_generator.state)

    def load_coverage_state_dict(self,state):
        if self.coverage_sampler is not None:
            self.coverage_sampler.load_state_dict(state['sampler'])
            self.coverage_rng.bit_generator.state=state['rng']

    @property
    def ranking_epochs(self):
        return self.coverage_sampler.passes if self.coverage_sampler is not None else None

    @staticmethod
    def _draw(groups,count,rng,max_groups=None):
        if not count:return np.empty(0,dtype=np.int64)
        if max_groups is not None and len(groups)>max_groups:
            groups=[groups[i] for i in rng.choice(len(groups),max_groups,replace=False)]
        which=rng.integers(len(groups),size=count)
        selected=np.concatenate([(BoardRelationSampler._draw(group,int((which==j).sum()),rng) if isinstance(group,list)
                                  else rng.choice(group,int((which==j).sum()))) for j,group in enumerate(groups)
                                 if (which==j).any()]) if count else np.empty(0,dtype=np.int64)
        rng.shuffle(selected); return selected

    def draw(self,source_sampler,point_count,rank_count,rng):
        nc=round(rank_count*self.coverage_fraction)
        nf=round(point_count*self.finite_fraction); nb=round((rank_count-nc)*self.boundary_fraction)
        p=source_sampler.draw(point_count-nf); broad=self.data['pairs_train'][p]
        broad=sample_goals(self.corpus,broad[:,0],rng,self.data['reserved_relations'],fallback=broad[:,1])
        limit=self.options.get('layouts_per_batch') if self.options.get('balance_layouts') else None
        finite=self.points[self._draw(self.point_groups,nf,rng,limit)]
        pair=np.concatenate([broad,finite]); p=np.concatenate([p,self.point_index[finite[:,0]]])
        selected=np.concatenate([self._draw(self.boundary_groups,nb,rng,limit),self._draw(self.other_groups,rank_count-nc-nb,rng,limit)])
        rng.shuffle(selected)
        if nc: selected=np.concatenate([self.coverage_sampler.draw(nc),selected])
        tasks=self.ranks[selected]
        return p,self.rank_index[tasks[:,0]],pair,tasks


def decision_neighborhood_points(pairs,indices,tasks,children,point_index,fraction,rng):
    """Replace part of point supervision by same-goal parent/child pairs.

    Main sampler RNG and all ranking decisions stay unchanged. Supplemental
    sources outside the original point bank use index -1 for its draw counter;
    they still receive exact labels and normal endpoint-exposure accounting.
    """
    if not 0<=fraction<=1: raise ValueError('Invalid neighborhood point fraction')
    count=round(len(pairs)*fraction/2)
    origin=np.zeros(len(pairs),dtype=np.uint8)
    if not count: return pairs,indices,origin
    chosen=rng.choice(len(tasks),count,replace=count>len(tasks))
    valid=children[chosen]>=0; lengths=valid.sum(axis=1)
    if (lengths==0).any(): raise ValueError('Ranking parent has no legal child')
    ordinal=rng.integers(lengths)
    slots=(valid & (valid.cumsum(axis=1)==ordinal[:,None]+1)).argmax(axis=1)
    parent=tasks[chosen]
    child=np.column_stack([children[chosen,slots],parent[:,1]])
    pairs=pairs.copy(); indices=indices.copy()
    pairs[-2*count:]=np.concatenate([parent,child])
    indices[-2*count:]=point_index[pairs[-2*count:,0]]
    origin[-2*count:-count]=1; origin[-count:]=2
    return pairs,indices,origin


class FailureReplaySampler:
    """Replay fixed training failures and their permitted optimal continuations.

    The caller supplies a separate checkpointed RNG. Every replayed decision
    also supplies point labels for its parent and ALL legal children, so the
    local branch is calibrated at one complete goal within the same batch.
    """
    def __init__(self,corpus,data,bank,options):
        self.corpus,self.options=corpus,options
        self.fraction=options['ranking_fraction']; self.critical_fraction=options['critical_fraction']
        if not 0<self.fraction<=1 or not 0<self.critical_fraction<1: raise ValueError('Invalid replay quotas')
        self.tasks,self.reason,exact=self.validate(corpus,data,bank)
        self.children=exact['children']; labels=exact['distances']
        edges=np.asarray(options['distance_bin_upper_bounds'],dtype=np.float32)
        if np.any(np.diff(edges)<=0) or edges[0]<0 or edges[-1]<labels.max(): raise ValueError('Replay distance bins omit a label')
        scenes=corpus.locate(self.tasks[:,0]); bins=np.searchsorted(edges,labels)
        def groups(mask):
            result=[]
            for sid in np.unique(scenes[mask]):
                take=(scenes==sid)&mask
                result.append([np.flatnonzero(take&(bins==b)) for b in np.unique(bins[take])])
            if not result: raise ValueError('Both critical and continuation replay contexts required')
            return result
        self.critical=groups((self.reason&15)>0);self.continuations=groups((self.reason&16)>0)
        self.summary=dict(decisions=len(self.tasks),critical_decisions=int(((self.reason&15)>0).sum()),
            continuation_decisions=int(((self.reason&16)>0).sum()),layouts=len(np.unique(scenes)),
            all_labels_exact=True,all_legal_candidates_preserved=True,training_boundaries_checked=True,options=options)

    @staticmethod
    def validate(corpus,data,bank):
        tasks=np.asarray(bank['tasks'],dtype=np.int64);reason=np.asarray(bank['reason'],dtype=np.uint8)
        if not len(tasks) or len(np.unique(tasks,axis=0))!=len(tasks):raise ValueError('Unique nonempty replay decisions required')
        if reason.shape!=(len(tasks),) or (reason==0).any() or (reason&np.uint8(224)).any():raise ValueError('Unknown replay reason')
        if not training_decision_mask(corpus,data,tasks).all():raise ValueError('Replay crosses protected training boundary')
        exact=materialize(corpus,tasks)
        for key in ('children','distances','child_distances','optimal','reachability'):
            if not np.array_equal(bank[key],exact[key]):raise ValueError(('Replay labels or legal candidates changed',key))
        return tasks,reason,exact

    def apply(self,pairs,point_indices,tasks,rank_indices,point_index,rank_index,rng):
        n=round(len(tasks)*self.fraction);nc=round(n*self.critical_fraction)
        limit=self.options.get('layouts_per_batch')
        ids=np.concatenate([BoardRelationSampler._draw(self.critical,nc,rng,limit),
                            BoardRelationSampler._draw(self.continuations,n-nc,rng,limit)])
        roles=np.concatenate([np.ones(nc,dtype=np.uint8),np.full(n-nc,2,dtype=np.uint8)])
        order=rng.permutation(n);ids,roles=ids[order],roles[order]
        return self.replace(pairs,point_indices,tasks,rank_indices,point_index,rank_index,ids,roles)

    def replace(self,pairs,point_indices,tasks,rank_indices,point_index,rank_index,ids,roles,point_positions=None):
        n=len(ids);origin=np.zeros(len(pairs),dtype=np.uint8);rank_origin=np.zeros(len(tasks),dtype=np.uint8)
        replay_ids=np.full(len(tasks),-1,dtype=np.int64)
        if not n:return pairs,point_indices,tasks,rank_indices,origin,rank_origin,replay_ids
        replay=self.tasks[ids];children=self.children[ids];valid=children>=0
        child_pairs=np.column_stack([children[valid],np.broadcast_to(replay[:,1,None],children.shape)[valid]])
        points=np.concatenate([replay,child_pairs]);np_=len(points)
        if np_>len(pairs): raise ValueError('Replay branches exceed point batch budget')
        positions=np.arange(len(pairs)-np_,len(pairs)) if point_positions is None else np.asarray(point_positions)
        if len(positions)!=np_ or len(np.unique(positions))!=np_ or (positions<0).any() or (positions>=len(pairs)).any():raise ValueError('Invalid replay point positions')
        pairs=pairs.copy();point_indices=point_indices.copy();tasks=tasks.copy();rank_indices=rank_indices.copy()
        pairs[positions]=points;point_indices[positions]=point_index[points[:,0]]
        origin[positions[:n]]=1;origin[positions[n:]]=2
        tasks[-n:]=replay;rank_indices[-n:]=rank_index[replay[:,0]]
        rank_origin[-n:]=roles;replay_ids[-n:]=ids
        return pairs,point_indices,tasks,rank_indices,origin,rank_origin,replay_ids


class LocalPathSampler(FailureReplaySampler):
    """Joint paths versus a degree/layout-matched shuffled control.

    Each complete block uses the identical context multiset in both modes.
    Shuffling only within layout and legal degree also preserves each batch's
    replay count, number of point labels and independent proposal prefixes.
    """
    def __init__(self,corpus,data,bank,options,rng,finite_point_fraction):
        from grounded_llm.core.qmap.sampling import GroupCoverageSampler
        self.corpus,self.options,self.rng=corpus,options,rng
        if not 0<finite_point_fraction<1:raise ValueError('Finite supplement fraction required')
        self.finite_point_fraction=finite_point_fraction
        self.tasks,self.reason,exact=self.validate(corpus,data,bank);self.children=exact['children']
        self.paths=np.asarray(bank['path_context_index'],dtype=np.int64);self.lengths=np.asarray(bank['path_lengths'],dtype=np.int64)
        if self.paths.ndim!=2 or self.paths.shape[1]!=options['maximum_steps'] or len(self.paths)!=len(self.lengths):raise ValueError('Path group shape changed')
        if (self.lengths<1).any() or (self.lengths>self.paths.shape[1]).any():raise ValueError('Invalid path length')
        valid=np.arange(self.paths.shape[1])[None,:]<self.lengths[:,None];ids=self.paths[valid]
        if (ids<0).any() or (ids>=len(self.tasks)).any() or (self.paths[~valid]!=-1).any():raise ValueError('Invalid path context index')
        for key,field in [('tasks','path_tasks'),('children','path_children'),('distances','path_distances'),('child_distances','path_child_distances'),('optimal','path_optimal')]:
            if not np.array_equal(bank[field][valid],exact[key][ids]):raise ValueError(('Path group labels changed',field))
        slots=bank['path_teacher_choice'][valid]
        if (slots<0).any() or (slots>=self.children.shape[1]).any() or not exact['optimal'][ids,slots].all():raise ValueError('Invalid optimal path action')
        for path,length in enumerate(self.lengths):
            contexts=self.paths[path,:length];rows=self.tasks[contexts];actions=bank['path_teacher_choice'][path,:length]
            following=self.children[contexts,actions]
            if not (rows[:,1]==rows[0,1]).all() or not np.array_equal(following[:-1],rows[1:,0]):raise ValueError('Path is not consecutive at one complete goal')
            reached=following[-1]==rows[0,1]
            if reached!=bool(bank['path_reached_goal'][path]) or (length<options['minimum_steps'] and not reached):raise ValueError('Incomplete short path group')
        if options['mode'] not in ('joint','shuffled') or options['groups_per_batch']<1 or options['block_batches']<1:raise ValueError('Invalid local path sampling options')
        self.scene_ids=corpus.locate(self.tasks[:,0]);self.degrees=(self.children>=0).sum(axis=1)
        path_scenes=self.scene_ids[self.paths[:,0]]
        self.group_sampler=GroupCoverageSampler(path_scenes,rng,options.get('layouts_per_batch'))
        self.queue=None;self.offsets=None;self.cursor=0;self.blocks=0;self.next_occurrence=0
        self.actual_path_steps=np.zeros(len(self.paths),dtype=np.uint32);self.last_annotations={}
        self.summary=dict(decisions=len(self.tasks),paths=len(self.paths),layouts=len(np.unique(path_scenes)),
            minimum_length=int(self.lengths.min()),maximum_length=int(self.lengths.max()),options=options,
            equal_context_multiset_per_complete_block=True,equal_layout_and_degree_per_batch=True,
            all_legal_candidates_preserved=True,all_labels_exact=True,training_boundaries_checked=True)

    def _plan(self):
        pieces=[];sizes=[]
        for _ in range(self.options['block_batches']):
            groups=self.group_sampler.draw(self.options['groups_per_batch']);parts=[]
            for group in groups:
                length=int(self.lengths[group]);occurrence=self.next_occurrence;self.next_occurrence+=1
                parts.append(np.column_stack([self.paths[group,:length],np.full(length,group),np.arange(length),np.full(length,occurrence)]))
            batch=np.concatenate(parts);pieces.append(batch);sizes.append(len(batch))
        planned=np.concatenate(pieces);contexts=planned[:,0]
        key=self.scene_ids[contexts]*(self.corpus.max_candidates+1)+self.degrees[contexts]
        order=np.argsort(key,kind='stable');_,start=np.unique(key[order],return_index=True)
        permutation=np.arange(len(planned))
        # Generate the same permutations even in joint mode, so the two modes
        # consume exactly the same separate RNG stream and future path plans.
        for positions in np.split(order,start[1:]):permutation[positions]=self.rng.permutation(positions)
        self.queue=planned if self.options['mode']=='joint' else planned[permutation]
        self.offsets=np.cumsum([0,*sizes]);self.cursor=0;self.blocks+=1

    def apply(self,pairs,point_indices,tasks,rank_indices,point_index,rank_index,rng):
        if rng is not self.rng:raise ValueError('Local path RNG identity changed')
        if self.queue is None or self.cursor==self.options['block_batches']:self._plan()
        start,stop=self.offsets[self.cursor:self.cursor+2];plan=self.queue[start:stop];self.cursor+=1
        ids=plan[:,0];n=len(ids)
        if n>len(tasks):raise ValueError('Whole path groups exceed ranking budget')
        self.last_annotations={key:np.full(len(tasks),-1,dtype=np.int64) for key in ['rank_path_id','rank_path_step','rank_path_occurrence','rank_path_block']}
        for column,key in enumerate(['rank_path_id','rank_path_step','rank_path_occurrence'],1):self.last_annotations[key][-n:]=plan[:,column]
        self.last_annotations['rank_path_block'][-n:]=self.blocks-1
        np.add.at(self.actual_path_steps,plan[:,1],1)
        count=n+int((self.children[ids]>=0).sum())
        prefix=len(pairs)-round(len(pairs)*self.finite_point_fraction)
        positions=np.arange(prefix,len(pairs));truth=self.corpus.distance(pairs[positions,0],pairs[positions,1])
        if not np.isfinite(truth).all():raise ValueError('Original finite supplement contains an unreachable pair')
        cutoff=self.options['preserve_point_distances_up_to']
        far=self.rng.permutation(positions[truth>cutoff])
        # Retain original positive near labels whenever the finite far pool
        # suffices. In a shortfall, protect distances1-2 until the last resort.
        middle=self.rng.permutation(positions[(truth>2)&(truth<=cutoff)])
        close=self.rng.permutation(positions[truth<=min(2,cutoff)])
        replacement=np.concatenate([far,middle,close])[:count]
        if len(replacement)!=count:raise ValueError('Whole paths exceed finite supplement budget')
        return self.replace(pairs,point_indices,tasks,rank_indices,point_index,rank_index,ids,np.full(n,2,dtype=np.uint8),replacement)

    def state_dict(self):
        return dict(paths=self.paths,group_sampler=self.group_sampler.state_dict(),queue=self.queue,offsets=self.offsets,
            cursor=self.cursor,blocks=self.blocks,next_occurrence=self.next_occurrence,actual_path_steps=self.actual_path_steps)

    def load_state_dict(self,state):
        if not np.array_equal(state['paths'],self.paths):raise ValueError('Path group identity changed during resume')
        self.group_sampler.load_state_dict(state['group_sampler'])
        for key in ['queue','offsets','cursor','blocks','next_occurrence','actual_path_steps']:setattr(self,key,state[key])


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
        endpoint_holdout=spec.get('holdout_goal_endpoints',True)
        safe_parent=~np.isin(np.arange(len(scene.states)),blocked) if endpoint_holdout else np.ones(len(scene.states),dtype=bool)
        safe_rank=safe_parent & ~np.isin(scene.children,blocked).any(axis=1) if endpoint_holdout else safe_parent.copy()
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
                if not endpoint_holdout:reserved.append(closure(corpus,selected))
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
        if spec.get('exhaustive_training_goals'):
            for j,g in enumerate(scene.goals[:nt]):
                parents=np.flatnonzero(safe_parent)+offset
                rows=np.column_stack([parents,np.full(len(parents),int(g)+offset)])
                rows=rows[~np.isin(keys(rows[:,0],rows[:,1],corpus.size),reserved)]
                pairs.append(rows)
                label=scene.distances[j]
                parents=np.flatnonzero(safe_rank & np.isfinite(label) & (label>0))+offset
                rows=np.column_stack([parents,np.full(len(parents),int(g)+offset)])
                if len(rows):
                    children=corpus.successors(parents);valid=children>=0
                    related=np.full(children.shape,-1,dtype=np.int64)
                    related[valid]=keys(children[valid],np.full(valid.sum(),int(g)+offset),corpus.size)
                    safe=~np.isin(keys(rows[:,0],rows[:,1],corpus.size),reserved)&~np.isin(related,reserved).any(axis=1)
                    decisions.append(rows[safe])
            continue
        parents=np.flatnonzero(safe_parent)+offset
        limit=spec.get('training_sources_per_scene')
        local_rng=np.random.default_rng(group_seed(json.dumps(scene.metadata['source'],sort_keys=True),config['seed'],'training_sources')) if limit else rng
        if limit and len(parents)>limit:
            nearest=np.min(scene.distances[:nt],axis=0)
            finite=np.flatnonzero(safe_parent & np.isfinite(nearest) & (nearest>0))+offset
            near=balanced(np.column_stack([finite,np.zeros(len(finite),dtype=np.int64)]),limit//2,nearest[finite-offset],local_rng)[:,0]
            chosen=np.unique(np.concatenate([local_rng.choice(parents,limit//2,replace=False),near]))
            rest=np.setdiff1d(parents,chosen)
            parents=np.sort(np.concatenate([chosen,local_rng.choice(rest,limit-len(chosen),replace=False)]))
        rows,_=sample_goals(corpus,parents,rng,reserved,allow_missing=True)
        pairs.append(rows)
        reachable=np.isfinite(scene.distances[:nt]).any(axis=0)
        positive=(scene.distances[:nt]>0).any(axis=0)
        parents=np.flatnonzero(safe_rank & reachable & positive)+offset
        if limit and len(parents)>limit:
            nearest=np.min(scene.distances[:nt],axis=0)
            chosen=np.unique(balanced(np.column_stack([parents,np.zeros(len(parents),dtype=np.int64)]),limit,nearest[parents-offset],local_rng)[:,0])
            rest=np.setdiff1d(parents,chosen)
            parents=np.sort(np.concatenate([chosen,local_rng.choice(rest,limit-len(chosen),replace=False)]))
        rows,_=sample_goals(corpus,parents,rng,reserved,ranking=True,allow_missing=True)
        decisions.append(rows)
    return {k:np.concatenate(v) for k,v in groups.items() if v},np.concatenate(pairs),np.concatenate(decisions),reserved,None


def token_cohorts(corpus,config,rng):
    if config['relations'].get('protocol')=='query_goal_holdout':
        return token_goal_cohorts(corpus,config,rng)
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


def token_goal_cohorts(corpus,config,rng):
    """Fit continuous branches while holding out queries, not state inputs."""
    spec=config['relations'];count=spec['evaluation_count']
    nt,nh=spec['training_goals'],spec['heldout_goals_per_split']
    chosen=rng.choice(corpus.size,nt+2*nh,replace=False)
    pools=[np.sort(chosen[:nt]),chosen[nt:nt+nh],chosen[nt+nh:]]
    corpus.training_goal_pool=pools[0]
    near_training=np.empty(0,dtype=np.int64)
    if spec.get('preserve_training_goal_neighborhood'):
        terminal=np.column_stack([corpus.successors(pools[0]).ravel(),np.repeat(pools[0],corpus.max_candidates)])
        near_training=closure(corpus,terminal)
    groups={};reserved=[]
    for name,pool in [('validation',pools[0]),('test',pools[0]),
                      ('unseen_goal_validation',pools[1]),('unseen_goal_test',pools[2])]:
        goals=rng.choice(pool,count*8);sources=rng.integers(corpus.size,size=len(goals))
        half=len(goals)//2;sources[:half]=goals[:half]
        lengths=rng.integers(1,2*int(corpus.metadata['statistics']['diameter'])+1,size=half)
        for step in range(int(lengths.max())):
            ids=np.flatnonzero(lengths>step)
            sources[ids]=corpus.children[sources[ids],rng.integers(corpus.max_candidates,size=len(ids))]
        rows=np.column_stack([sources,goals]);rows=rows[rows[:,0]!=rows[:,1]]
        if name in ('validation','test') and spec.get('minimum_relation_holdout_distance',0):
            rows=rows[corpus.distance(rows[:,0],rows[:,1])>=spec['minimum_relation_holdout_distance']]
        if reserved or len(near_training):
            protected=np.unique(np.concatenate([near_training,*reserved]));children=corpus.successors(rows[:,0])
            safe=~np.isin(keys(rows[:,0],rows[:,1],corpus.size),protected)
            safe&=~np.isin(keys(children,rows[:,1,None],corpus.size),protected).any(axis=1);rows=rows[safe]
        selected=balanced(rows,count,corpus.distance(rows[:,0],rows[:,1]),rng)
        groups[name]=selected;reserved.append(closure(corpus,selected))
    reserved=np.unique(np.concatenate(reserved));sources=np.arange(corpus.size,dtype=np.int64)
    pairs=sample_goals(corpus,sources,rng,reserved,allowed=pools[0])
    decisions=sample_goals(corpus,sources,rng,reserved,ranking=True,allowed=pools[0])
    return groups,pairs,decisions,reserved,np.zeros(corpus.size,dtype=np.int8)


def prepare(config,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    module=importlib.import_module(f'grounded_llm.tasks.{config["task"]}.relation_data')
    corpus=module.load(out/'graphs') if (out/'graphs'/'corpus.json').exists() or (out/'graphs'/'oracle.json').exists() else module.build(config,out/'graphs')
    rng=np.random.default_rng(config['seed']+1)
    groups,pairs,decisions,reserved,partition=(token_cohorts if config['task']=='token_swapping' else board_cohorts)(corpus,config,rng)
    data={'pairs_train':pairs,'pair_distances_train':corpus.distance(pairs[:,0],pairs[:,1]),'reserved_relations':reserved,
          'holdout_goal_endpoints':np.array(config['relations'].get('holdout_goal_endpoints',True))}
    if hasattr(corpus,'training_goal_pool'):data['training_goal_pool']=corpus.training_goal_pool
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
    if config['task']=='game24':
        roots=np.asarray([corpus.offsets[i]+s.metadata['original_state'] for i,s in enumerate(corpus.scenes) if s.metadata['split']=='train'])
        summary['training_original_starts']=len(roots)
        summary['original_starts_with_candidate_supervision']=int(np.isin(roots,decisions[:,0]).sum())
        summary['goal_holdout_scope']='endpoints' if bool(data['holdout_goal_endpoints']) else 'query_goal_relations; candidate endpoint exposure allowed'
    if 'training_goal_pool' in data:
        summary['training_goals']=len(data['training_goal_pool'])
        summary['goal_holdout_scope']='query_goal_relations; all physical state inputs allowed'
    atomic_json(out/'summary.json',summary)
    return corpus,data,summary


def load(config,out):
    out=Path(out)
    extra=dict(feature_encoding=config.get('input_encoding','identity')) if config['task'] in ('rush_hour','bridge') else {}
    corpus=importlib.import_module(f'grounded_llm.tasks.{config["task"]}.relation_data').load(out/'graphs',**extra)
    data=dict(np.load(out/'relations.npz'))
    return corpus,data,json.loads((out/'summary.json').read_text())
