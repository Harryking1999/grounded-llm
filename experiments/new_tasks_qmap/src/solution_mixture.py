"""Original broad relationships plus complete target graphs; fixed exact losses."""
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from .relation_data import keys
from .solution_structure import context_points


def offsets(records):
    return np.cumsum([0] + [row['states'] for row in records], dtype=np.int64)


def remap_ids(values, small_records, full_records):
    """State identity is (physical scene path, local state), never list position."""
    old = offsets(small_records); new = offsets(full_records)
    lookup = {row['path']:i for i, row in enumerate(full_records)}
    destination = []
    for row in small_records:
        sid = lookup[row['path']]
        if row['states'] != full_records[sid]['states'] or row['initial_board'] != full_records[sid]['initial_board']:
            raise ValueError('Physical scene identity changed')
        destination.append(sid)
    values = np.asarray(values); valid = values >= 0; result = values.copy()
    if np.any(values[valid] >= old[-1]): raise ValueError('State outside source corpus')
    sid = np.searchsorted(old[1:], values[valid], side='right')
    result[valid] = values[valid] - old[sid] + new[np.asarray(destination)[sid]]
    return result


def contains(sorted_keys, values):
    values = np.asarray(values)
    if not len(sorted_keys): return np.zeros(values.shape, bool)
    index = np.searchsorted(sorted_keys, values)
    return (index < len(sorted_keys)) & (sorted_keys[np.minimum(index, len(sorted_keys)-1)] == values)


def protected_mask(data, reserved, size, ranking):
    rows = data['tasks_train' if ranking else 'pairs_train']
    safe = np.ones(len(rows), bool)
    for start in range(0, len(rows), 262144):
        stop = min(start+262144, len(rows)); batch = rows[start:stop]
        blocked = contains(reserved, keys(batch[:, 0], batch[:, 1], size))
        if ranking:
            children = data['children_train'][start:stop]; valid = children >= 0
            candidate_keys = np.full(children.shape, -1, np.int64)
            candidate_keys[valid] = keys(children[valid], np.broadcast_to(batch[:, 1, None], children.shape)[valid], size)
            blocked |= contains(reserved, candidate_keys).any(axis=1)
        safe[start:stop] = ~blocked
    return safe


def combine_banks(broad, graph, full_records, small_records, extra_reserved=()):
    """Union guards first; reject whole decisions, preserving all action slots."""
    size = int(offsets(full_records)[-1]); small_size = int(offsets(small_records)[-1])
    mapped = {}
    for name, values in graph.items():
        if name.startswith(('tasks_', 'children_', 'solution_paths_')) or name == 'pairs_train':
            mapped[name] = remap_ids(values, small_records, full_records)
        elif name not in ('reserved_relations', 'terminal_ids', 'auxiliary_ids', 'feedback_goal_weights'):
            mapped[name] = values
    old_keys = graph['reserved_relations']
    endpoints = remap_ids(np.column_stack([old_keys//small_size, old_keys%small_size]), small_records, full_records)
    reserved = np.unique(np.concatenate([broad['reserved_relations'], keys(endpoints[:, 0], endpoints[:, 1], size),
                                         np.asarray(extra_reserved,dtype=np.int64)]))
    # A held-out intermediate target stays absent from BOTH training pools as
    # a target. Seeing its state as an ordinary physical input is not a label.
    held = [v[:, 1] for k, v in mapped.items()
            if k.startswith('tasks_') and k not in ('tasks_train', 'tasks_fit_validation')]
    held_goals = np.unique(np.concatenate(held)) if held else np.empty(0,np.int64)
    accepted = []; audit = {}
    for name, bank in [('broad', broad), ('graph', mapped)]:
        p = protected_mask(bank, reserved, size, False) & ~np.isin(bank['pairs_train'][:, 1], held_goals)
        r = protected_mask(bank, reserved, size, True) & ~np.isin(bank['tasks_train'][:, 1], held_goals)
        audit[name] = dict(points_before=len(p), points_removed=int((~p).sum()),
                           decisions_before=len(r), decisions_removed=int((~r).sum()))
        accepted.append({k:v[p if k in ('pairs_train','pair_distances_train') else r]
                         for k,v in bank.items() if k.endswith('_train')})
    fields = ['pairs_train','pair_distances_train','tasks_train','distances_train',
              'children_train','child_distances_train','reachability_train','optimal_train']
    result = {k:np.concatenate([b[k] for b in accepted]) for k in fields}
    result.update({k:v for k,v in mapped.items() if not k.endswith('_train')})
    result['reserved_relations'] = reserved
    result['broad_point_count'] = np.asarray(len(accepted[0]['pairs_train']), np.int64)
    result['broad_rank_count'] = np.asarray(len(accepted[0]['tasks_train']), np.int64)
    for name, bank in zip(('broad','graph'), accepted):
        if not protected_mask(bank, reserved, size, False).all() or not protected_mask(bank, reserved, size, True).all():
            raise ValueError('Protected label survived the union filter')
        scenes = np.searchsorted(offsets(full_records)[1:], bank['tasks_train'][:, 0], side='right')
        point_scenes = np.searchsorted(offsets(full_records)[1:], bank['pairs_train'][:, 0], side='right')
        goal_scenes = np.searchsorted(offsets(full_records)[1:], bank['pairs_train'][:, 1], side='right')
        if not (scenes.size and all(full_records[i]['split']=='train' for i in np.unique(np.r_[scenes,point_scenes]))):
            raise ValueError('Nontraining physical layout entered training')
        if not np.array_equal(point_scenes, goal_scenes): raise ValueError('Cross-layout point label')
        if not np.array_equal(scenes,np.searchsorted(offsets(full_records)[1:],bank['tasks_train'][:,1],side='right')):
            raise ValueError('Cross-layout ranking context')
        audit[name]['training_layouts'] = len(np.unique(scenes))
        audit[name]['protected_point_collisions_after'] = 0
        audit[name]['protected_decision_collisions_after'] = 0
        audit[name]['held_target_collisions_after'] = int(np.isin(bank['tasks_train'][:,1], held_goals).sum())
    audit['reserved_relations'] = len(reserved)
    return result, audit


def grouped_rows(scene_ids, distances, eligible):
    """Build layout -> true distance strata without repeatedly scanning a bank."""
    ids = np.flatnonzero(eligible)
    bins = np.searchsorted([5, 10, 20, 32, np.inf], distances[ids])
    group = scene_ids[ids] * 8 + bins
    order = np.argsort(group, kind='stable'); ids = ids[order]; group = group[order]
    result = {}
    for part in np.split(np.arange(len(ids)), np.flatnonzero(np.diff(group))+1):
        if len(part): result.setdefault(int(group[part[0]]//8), []).append(ids[part])
    return list(result.values())


class MixtureSampler:
    """Broad layout balance; graph layout/target/exact-layer neighborhoods."""
    def __init__(self, corpus, data, options, rng):
        from .solution_graph_sampling import ConnectedGraphSampler
        self.corpus,self.data,self.options,self.rng = corpus,data,options,rng
        self.fraction = float(options['graph_fraction'])
        if not 0 <= self.fraction <= 1: raise ValueError('Invalid graph fraction')
        self.np,self.nr = int(data['broad_point_count']),int(data['broad_rank_count'])
        self.graph = None
        if self.fraction:
            view={k:v[self.np if k in ('pairs_train','pair_distances_train') else self.nr:]
                  for k,v in data.items() if k.endswith('_train')}
            self.graph=ConnectedGraphSampler(corpus,view,options,rng)
            # Every exact finite layer is eligible, including distances beyond
            # the inference budget. No errors or validation outcomes set weights.
            for goal,rows in self.graph.rows.items():
                labels=view['distances_train'][rows]
                self.graph.strata[goal]=[rows[labels==d] for d in np.unique(labels)]
        self.broad = None
        if self.fraction < 1:
            points=data['pair_distances_train'][:self.np]; ranks=data['distances_train'][:self.nr]
            ps=corpus.locate(data['pairs_train'][:self.np,0]);rs=corpus.locate(data['tasks_train'][:self.nr,0])
            boundary=((data['children_train'][:self.nr]>=0)&~np.isfinite(data['child_distances_train'][:self.nr])).any(axis=1)
            self.broad=dict(points=grouped_rows(ps,points,np.ones(len(points),bool)),
                finite=grouped_rows(ps,points,np.isfinite(points)&(points>0)),
                boundary=grouped_rows(rs,ranks,boundary),ordinary=grouped_rows(rs,ranks,~boundary))
            if any(not value for value in self.broad.values()): raise ValueError('Empty original sampling stratum')

    def draw(self, point_count, rank_count):
        from .relation_data import BoardRelationSampler
        gp,gr=round(point_count*self.fraction),round(rank_count*self.fraction)
        if bool(gp)!=bool(gr): raise ValueError('Mixture quota loses a supervision type')
        points=[];ranks=[]
        if gp:
            # ConnectedGraphSampler can optionally choose every root layer with
            # equal probability; this avoids its historical modulo-row bias.
            p,r=self.graph.draw(gp,gr);points.append(p+self.np);ranks.append(r+self.nr)
        if point_count-gp:
            pcount,rcount=point_count-gp,rank_count-gr
            finite=round(pcount*self.options.get('finite_point_fraction',.75))
            boundary=round(rcount*self.options.get('boundary_rank_fraction',.5))
            draw=lambda name,n:BoardRelationSampler._draw(self.broad[name],n,self.rng,self.options.get('broad_layouts_per_batch',32))
            points.extend([draw('points',pcount-finite),draw('finite',finite)])
            ranks.extend([draw('boundary',boundary),draw('ordinary',rcount-boundary)])
        p=np.concatenate(points);r=np.concatenate(ranks);self.rng.shuffle(p);self.rng.shuffle(r)
        return p,r


def prepare_mixture(config, out):
    from .solution_structure import load_corpus
    options=config['data'];source=Path(options['mixture_source']).resolve()
    for name in ['scenes.json','summary.json','relations.npz','feature_ids.npy','features_packed.npy']:
        (out/name).symlink_to(source/name)
    return load_corpus(config,out),dict(np.load(out/'relations.npz')),json.loads((out/'summary.json').read_text())


def broad_goal_bank(corpus, validation, options):
    """Fixed ordinary targets on validation layouts, entirely read-only."""
    from concurrent.futures import ThreadPoolExecutor
    scene_ids=corpus.locate(validation['tasks'][:,0])
    def prepare_scene(sid):
        rng=np.random.default_rng(options['broad_diagnostic_seed']+int(sid))
        scene=corpus.scenes[sid];offset=corpus.offsets[sid]
        if scene.metadata['split']!='validation':raise ValueError('Diagnostic layout is not validation')
        pool=np.unique(validation['tasks'][scene_ids==sid,0]);pool=pool[~np.isin(pool,scene.goals+offset)]
        rows=[];truth=[];accepted=[]
        states=np.arange(offset,corpus.offsets[sid+1])
        for goal in rng.permutation(pool):
            labels=corpus.distance(states,np.full(len(states),goal))
            if (np.isfinite(labels)&(labels>0)).sum()<32:continue
            accepted.append(int(goal))
            for lo,hi in [(0,1),(1,2),(2,5),(5,10),(10,20),(20,32),(32,64),(64,np.inf)]:
                candidates=np.flatnonzero(np.isfinite(labels)&(labels>lo)&(labels<=hi))
                chosen=rng.choice(candidates,min(options['broad_diagnostic_per_layer'],len(candidates)),replace=False)
                rows.extend((int(states[i]),int(goal)) for i in chosen);truth.extend(labels[chosen])
            if len(accepted)==options['broad_diagnostic_goals']:break
        if len(accepted)!=options['broad_diagnostic_goals']:raise ValueError(('Insufficient held-out ordinary targets',int(sid)))
        return rows,truth,accepted
    with ThreadPoolExecutor(max_workers=options['scene_loader_workers']) as executor:
        parts=list(executor.map(prepare_scene,np.unique(scene_ids)))
    rows=np.asarray([row for part in parts for row in part[0]],dtype=np.int64)
    truth=np.asarray([d for part in parts for d in part[1]],dtype=np.float32)
    return dict(tasks_broad_goal_validation=rows,distances_broad_goal_validation=truth),dict(
        layouts=len(parts),targets=sum(len(p[2]) for p in parts),queries=len(rows),
        scope='Fixed ordinary intermediate targets on the original 100 validation layouts. Geometry only, no model inspection. All exact labels and candidate closures remain outside both training pools.')


def prepare_joint(config, out):
    """Prepare one immutable joint bank shared by every matched proportion."""
    from types import SimpleNamespace
    from .solution_structure import load_corpus,exact_path
    from .relation_data import materialize
    from .relation_evaluate import planning_ids
    options=config['data'];broad_source=Path(options['broad_source']).resolve()
    graph_source=Path(options['graph_source']).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    full=json.loads((broad_source/'graphs/corpus.json').read_text())['scenes']
    small=json.loads((graph_source/'scenes.json').read_text())['scenes']
    with np.load(broad_source/'relations.npz') as archive:
        broad={k:archive[k] for k in archive.files if k.endswith('_train') or k=='reserved_relations'}
        validation={k:archive[k+'_unseen_layout_validation'] for k in
                    ['tasks','distances','children','child_distances','reachability','optimal']}
    graph=dict(np.load(graph_source/'relations.npz'))
    dense=dict(np.load(options['dense_validation_bank']))
    atomic_json(out/'scenes.json',dict(graphs=str(broad_source/'graphs'),scenes=full))
    corpus=load_corpus(config,out)
    broad_dense,broad_dense_scope=broad_goal_bank(corpus,validation,options)
    small_corpus=load_corpus(config,graph_source);extra=[]
    for name,rows in dense.items():
        if not name.startswith('tasks_'):continue
        children=small_corpus.successors(rows[:,0]);cp=context_points(rows,children)
        mapped=remap_ids(cp,small,full);extra.append(keys(mapped[:,0],mapped[:,1],int(offsets(full)[-1])))
    eligible=np.flatnonzero(np.isfinite(validation['distances'])&(validation['distances']>0)&(validation['distances']<=config['evaluation']['horizon']))
    validation={k:v[eligible] for k,v in validation.items()}
    ends=offsets(full)[1:]
    meta=SimpleNamespace(scenes=full,locate=lambda ids:np.searchsorted(ends,ids,side='right'))
    selected=planning_ids(meta,validation['tasks'],validation['distances'],config['evaluation']['rollout_count'],
                          np.random.default_rng(options['validation_seed']),config['evaluation']['planning_distance_bins'])
    vr=validation['tasks'][selected];vc=validation['children'][selected]
    # Unseen physical layouts were already disjoint. Include their relation
    # closures explicitly as well before checking either training pool.
    cp=context_points(vr,vc);extra.append(keys(cp[:,0],cp[:,1],int(ends[-1])))
    br=broad_dense['tasks_broad_goal_validation'];cp=context_points(br,corpus.successors(br[:,0]))
    extra.append(keys(cp[:,0],cp[:,1],int(ends[-1])))
    data,audit=combine_banks(broad,graph,full,small,np.concatenate(extra))
    if audit['broad']['training_layouts'] != options['required_broad_layouts']:
        raise ValueError('Original layout distribution was narrowed')
    for name,values in validation.items():data[name+'_broad_layout_validation']=values[selected]
    keep=broad_dense['distances_broad_goal_validation']<=config['evaluation']['horizon']
    gr=broad_dense['tasks_broad_goal_validation'][keep];gt=broad_dense['distances_broad_goal_validation'][keep]
    gi=planning_ids(meta,gr,gt,config['evaluation']['rollout_count'],np.random.default_rng(options['broad_diagnostic_seed']),
                    config['evaluation']['planning_distance_bins'])
    for name,values in materialize(corpus,gr[gi]).items():data[name+'_broad_goal_validation']=values
    paths=np.full((len(gi),config['evaluation']['horizon']+1),-1,np.int64)
    for i,(start,goal) in enumerate(gr[gi]):
        path=exact_path(corpus,start,goal,config['evaluation']['horizon']);paths[i,:len(path)]=path
    data['solution_paths_broad_goal_validation']=paths
    audit['broad_validation']=dict(queries=len(vr),layouts=len(np.unique(meta.locate(vr[:,0]))),
        scope='Previously used development validation layouts, original exact relations, fixed queries; never added to training.')
    # Both relation pools and every legal child keep their original exact labels.
    np.savez_compressed(out/'relations.npz',**data)
    for name,values in dense.items():
        if name.startswith('tasks_'):dense[name]=remap_ids(values,small,full)
    dense.update(broad_dense)
    np.savez_compressed(out/'dense_validation.npz',**dense)
    finite=np.r_[data['distances_train'],data['pair_distances_train']]
    summary=dict(task='sokoban',finite_max=float(finite[np.isfinite(finite)].max()),audit=audit,broad_goal_diagnostic=broad_dense_scope,
        broad_source=str(broad_source),graph_source=str(graph_source),
        point_pairs=len(data['pairs_train']),contexts=len(data['tasks_train']),
        sampling_scope='Fixed original 1000-layout prepared relations plus permitted complete target basins. No online goal replacement, validation feedback, isolated error oversampling or loss change in any arm.')
    atomic_json(out/'summary.json',summary);print(json.dumps(dict(stage='joint_bank',**summary)),flush=True)
    valid=data['children_train']>=0
    ids=np.unique(np.r_[data['pairs_train'].ravel(),data['tasks_train'][:,1],data['children_train'][valid]])
    np.save(out/'feature_ids.npy',ids)
    packed=np.lib.format.open_memmap(out/'features_packed.npy',mode='w+',dtype=np.uint8,
                                    shape=(len(ids),(corpus.input_dim+7)//8))
    for start in range(0,len(ids),65536):
        stop=min(start+65536,len(ids));values=corpus.features(ids[start:stop])
        if not ((values==0)|(values==1)).all():raise ValueError('Binary feature cache is not lossless')
        packed[start:stop]=np.packbits(values.astype(np.uint8),axis=1,bitorder='little')
        if start%(65536*16)==0:print(json.dumps(dict(stage='features',complete=stop,total=len(ids))),flush=True)
    packed.flush()
    probe=np.random.default_rng(32).choice(len(ids),min(512,len(ids)),replace=False)
    np.testing.assert_array_equal(np.unpackbits(packed[probe],axis=1,count=corpus.input_dim,bitorder='little'),corpus.features(ids[probe]))
    summary['lossless_cached_states']=len(ids);summary['feature_cache_bytes']=int(packed.nbytes)
    atomic_json(out/'summary.json',summary);atomic_json(out/'ready.json',dict(ready=True));print('ready',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare_joint(json.loads(args.config.read_text()),args.out)
