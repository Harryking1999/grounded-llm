"""State-pair mode of the existing new_tasks_qmap entry; one task per invocation."""
import argparse
import json
from pathlib import Path
import time
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.core.files import atomic_json, atomic_save
from grounded_llm.core.qmap.model import StateEncoder, soft_rank_loss
from grounded_llm.core.qmap.exact_graph import edge_costs
from grounded_llm.core.qmap.sampling import StratifiedEpochSampler, CoverageSampler, GroupCoverageSampler
from .relation_data import prepare, load, sample_goals, BoardRelationSampler, decision_neighborhood_points, FailureReplaySampler, LocalPathSampler
from .relation_evaluate import evaluate


def learning_rate_at_epoch(options,epoch):
    rate=options['learning_rate']
    schedule=options.get('learning_rate_schedule',[])
    if any(b['start_epoch']<=a['start_epoch'] for a,b in zip(schedule,schedule[1:])):
        raise ValueError('Learning-rate stages must have increasing start epochs')
    for stage in schedule:
        if stage['learning_rate']<=0 or stage['start_epoch']<0: raise ValueError('Invalid learning-rate stage')
        if epoch>=stage['start_epoch']: rate=stage['learning_rate']
    return rate


@torch.no_grad()
def training_probe(model,corpus,batches,options):
    """Fixed relations actually used by the optimizer; never select weights here."""
    if not batches: return None
    from .relation_evaluate import QCache,safe_scores
    model.eval();cache=QCache(model,corpus,next(model.parameters()).device)
    pairs=np.concatenate([b['pairs'] for b in batches]);truth=np.concatenate([b['point_truth'] for b in batches])
    tasks=np.concatenate([b['tasks'] for b in batches]);children=np.concatenate([b['children'] for b in batches])
    child_truth=np.concatenate([b['child_truth'] for b in batches]);valid=children>=0
    predictions=cache.scores(pairs[:,0],pairs[:,1]);finite=np.isfinite(truth)
    target=np.where(finite,truth,options['unreachable_target'])
    scores=safe_scores(cache,children,tasks[:,1]);device=next(model.parameters()).device
    costs=edge_costs(corpus,tasks[:,0])
    ranking=soft_rank_loss(torch.as_tensor(np.where(valid,costs+scores,0),device=device),
        torch.as_tensor(costs+np.where(np.isfinite(child_truth),child_truth,options['unreachable_target']),device=device),
        torch.as_tensor(valid,device=device),options['rank_temperature'])
    parent=corpus.distance(tasks[:,0],tasks[:,1]);eligible=np.isfinite(parent)&(parent>0)
    chosen=np.where(valid,costs+scores,np.inf).argmin(axis=1)
    optimal=(child_truth+costs)[np.arange(len(tasks)),chosen]==parent
    error=np.abs(predictions-target)
    return dict(batches=len(batches),point_count=len(pairs),decision_count=len(tasks),
        finite_mae=float(np.abs(predictions[finite]-truth[finite]).mean()) if finite.any() else None,
        distance_huber=float(np.where(error<1,.5*error**2,error-.5).mean()),candidate_soft_rank_loss=float(ranking),
        shortest_action_rate=float(optimal[eligible].mean()) if eligible.any() else None,
        by_distance={name:dict(count=int(mask.sum()),mae=float(np.abs(predictions[mask]-truth[mask]).mean()))
            for name,mask in [('1_5',finite&(truth>=1)&(truth<=5)),('6_20',finite&(truth>=6)&(truth<=20)),('21_plus',finite&(truth>=21))] if mask.any()},
        role='Fixed actual training batches; fitting diagnostic only, never checkpoint selection.')


def scope(config,corpus):
    if config['task']=='token_swapping': return corpus.scope
    if config['task']=='bridge': return dict(people=corpus.scenes[0].env.n,complete_goal=True,weighted=True)
    if config['task']=='game24': return dict(max_numbers=corpus.scenes[0].env.max_numbers,complete_goal=True,exact_rationals=True)
    return dict(board=[10,10] if config['task']=='sokoban' else [6,6],
                boxes=4 if config['task']=='sokoban' else None,wall_count=0 if config['task']=='rush_hour' else None,
                complete_goal=True,shared_across_layouts=True)


def validate_continuation(previous,current,step):
    """Authorize only a larger update budget; retain every training condition."""
    old,new=deepcopy(previous),deepcopy(current)
    for item in (old,new):
        for key in ('continuation','convergence','pilot'):item.pop(key,None)
    if not step<=old['training']['steps']<new['training']['steps']:
        raise ValueError('Continuation must extend the original update budget')
    new['training']['steps']=old['training']['steps']
    if old!=new:raise ValueError('Budget continuation changes data, sampler, model, loss, optimizer or evaluation')


def retain_checkpoint(source,target):
    """Preserve a full atomic checkpoint when last.pt is subsequently replaced."""
    import os,shutil
    source,target=Path(source),Path(target)
    temporary=target.with_name(target.name+'.link')
    try:os.link(source,temporary)
    except OSError:shutil.copy2(source,temporary)
    temporary.replace(target)


def train(config,out,source_commit,device,resume=False,prepare_only=False):
    out=Path(out)
    if out.exists() and not resume:
        raise FileExistsError('Existing exact-pair run; use --resume with its original config')
    out.mkdir(parents=True,exist_ok=True)
    if resume and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Resume requires the unchanged formal configuration')
    atomic_json(out/'config.json',config)
    structure_options=config['training'].get('solution_structure')
    reference=config.get('data',{}).get('prepared_from') if not structure_options else None
    if reference and not (out/'data').exists():
        source=(out/reference).resolve()
        previous=json.loads((source.parent/'config.json').read_text())
        if any(previous[key]!=config[key] for key in ('task','seed','relations')):
            raise ValueError('Prepared relation reference changes task, seed or fixed splits')
        if {k:v for k,v in previous['data'].items() if k!='prepared_from'}!={k:v for k,v in config['data'].items() if k!='prepared_from'}:
            raise ValueError('Prepared relation reference changes physical data scope')
        (out/'data').symlink_to(Path(reference),target_is_directory=True)
    prepared=time.monotonic()
    if structure_options:
        from .solution_structure import prepare as prepare_structure
        corpus,data,description=prepare_structure(config,out/'data')
    else:
        corpus,data,description=(load(config,out/'data') if (out/'data'/'relations.npz').exists() else prepare(config,out/'data'))
    preparation_seconds=time.monotonic()-prepared
    if not (out/'preparation.json').exists(): atomic_json(out/'preparation.json',dict(seconds=preparation_seconds,data=description))
    preparation_seconds=json.loads((out/'preparation.json').read_text())['seconds']
    if prepare_only:
        print(json.dumps(dict(prepared=str(out),data=description)),flush=True)
        return
    options=config['training']
    if description.get('finite_max',0)>=min(config['model']['cap'],options['unreachable_target']):
        raise ValueError('Prepared finite oracle costs exceed the configured encoding scale')
    torch.set_num_threads(options['cpu_threads']); torch.manual_seed(config['seed'])
    rng=np.random.default_rng(config['seed']+options.get('sampling_seed_offset',2))
    neighborhood_rng=np.random.default_rng(config['seed']+2003)
    replay_rng=np.random.default_rng(config['seed']+62003)
    model=StateEncoder(input_dim=corpus.input_dim,**config['model']).to(device)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=options['learning_rate'])
    point_truth=data['pair_distances_train']
    point_labels=np.where(np.isfinite(point_truth),point_truth,options['unreachable_target']).astype(np.float32)
    relation_sampler=(BoardRelationSampler(corpus,data,options['relation_sampling']) if options.get('relation_sampling') else None)
    if options.get('ranking_coverage_epochs'):
        if relation_sampler is None or relation_sampler.coverage_sampler is None:
            raise ValueError('Epoch budget requires an explicit ranking coverage traversal')
        per_step=round(options['decision_batch_size']*relation_sampler.coverage_fraction)
        expected=int(np.ceil(len(relation_sampler.ranks)*options['ranking_coverage_epochs']/per_step))
        if options['steps']!=expected: raise ValueError(('Step budget does not match ranking epochs',expected))
    if options.get('learning_rate_schedule') and (relation_sampler is None or relation_sampler.coverage_sampler is None):
        raise ValueError('Epoch learning-rate schedule requires ranking coverage')
    neighborhood_fraction=options.get('decision_neighborhood_point_fraction',0.)
    if neighborhood_fraction and (relation_sampler is None or neighborhood_fraction>relation_sampler.finite_fraction):
        raise ValueError('Neighborhood points must replace the finite supplement, preserving broad coverage')
    if relation_sampler: atomic_json(out/'sampling_summary.json',relation_sampler.summary)
    replay_sampler=None;replay_draws=None
    if options.get('failure_replay'):
        if relation_sampler is not None and relation_sampler.coverage_sampler is not None:
            raise ValueError('Replay would overwrite coverage rows; use separate conditions')
        replay_options=options['failure_replay']
        if relation_sampler is None or neighborhood_fraction:
            raise ValueError('Failure replay requires board sampling and its own complete point neighborhoods')
        local_paths=replay_options.get('sampler')=='local_paths'
        if local_paths and options['steps']%replay_options['block_batches']:
            raise ValueError('Matched local-path budget must contain complete sampling blocks')
        maximum_contexts=(replay_options['groups_per_batch']*replay_options['maximum_steps'] if local_paths
                          else round(options['decision_batch_size']*replay_options['ranking_fraction']))
        maximum_replay_points=maximum_contexts*(corpus.max_candidates+1)
        if maximum_contexts>options['decision_batch_size']:raise ValueError('Replay exceeds ranking budget')
        if maximum_replay_points>round(options['batch_size']*relation_sampler.finite_fraction):
            raise ValueError('Replay would replace broad point coverage')
        source=(out/replay_options['bank']).resolve();snapshot=out/'failure_replay_bank.npz'
        if not snapshot.exists():
            import shutil
            temporary=snapshot.with_suffix('.tmp');shutil.copyfile(source,temporary);temporary.replace(snapshot)
        with np.load(snapshot) as archive: bank={key:archive[key] for key in archive.files}
        replay_sampler=(LocalPathSampler(corpus,data,bank,replay_options,replay_rng,relation_sampler.finite_fraction) if local_paths
                        else FailureReplaySampler(corpus,data,bank,replay_options))
        replay_draws=np.zeros(len(replay_sampler.tasks),dtype=np.uint32)
        atomic_json(out/'failure_replay_summary.json',replay_sampler.summary)
    def sampler(labels,broad_only=False):
        if options.get('broad_fraction') or broad_only:
            return CoverageSampler(labels,rng,len(labels),1. if broad_only else options['broad_fraction'])
        return StratifiedEpochSampler(labels,rng)
    pair_sampler=(GroupCoverageSampler(corpus.locate(data['pairs_train'][:,0]),rng,options.get('layouts_per_batch'))
                  if options.get('balance_layouts') else sampler(point_labels,broad_only=bool(relation_sampler)))
    rank_sampler=(GroupCoverageSampler(corpus.locate(data['tasks_train'][:,0]),rng,options.get('layouts_per_batch'))
                  if options.get('balance_rank_layouts') else sampler(data['distances_train']))
    solution_sampler=None
    if structure_options:
        from .solution_structure import SolutionSampler
        if structure_options['mode']=='connected_graphs':
            from .solution_graph_sampling import ConnectedGraphSampler
            SolutionSampler=ConnectedGraphSampler
        elif structure_options['mode']=='broad_graph_mixture':
            from .solution_mixture import MixtureSampler
            SolutionSampler=MixtureSampler
        if relation_sampler or options.get('online_goals') or replay_sampler:
            raise ValueError('Solution pilot uses fixed exact banks and its own sampling, without online replacement')
        solution_sampler=SolutionSampler(corpus,data,structure_options,rng)
    exposure=np.zeros(corpus.size,dtype=np.uint32)
    pair_draws=np.zeros(len(point_truth),dtype=np.uint32); rank_draws=np.zeros(len(data['tasks_train']),dtype=np.uint32)
    allowed=None
    if options.get('online_goals') and not hasattr(corpus,'scenes'):
        allowed=np.flatnonzero(np.load(out/'data'/'endpoint_partition.npy')==0)
    edge_bank=None
    if options.get('action_edge_fraction',0):
        fraction=options['action_edge_fraction']
        if not 0<fraction<1 or config['task']=='bridge':raise ValueError('Direct unit-edge quota requires a unit-cost task')
        from .relation_data import keys
        if hasattr(corpus,'scenes'):
            banks=[]
            for sid,scene in enumerate(corpus.scenes):
                if scene.metadata['split']!='train':continue
                parent,slot=np.nonzero(scene.children>=0);child=scene.children[parent,slot]
                held=scene.goals[scene.metadata['training_goals']:]
                safe=~np.isin(parent,held)&~np.isin(child,held)
                banks.append(np.column_stack([parent[safe]+corpus.offsets[sid],child[safe]+corpus.offsets[sid]]))
            edge_bank=np.concatenate(banks)
        else:
            partition=np.load(out/'data'/'endpoint_partition.npy')
            parent,slot=np.nonzero((partition[:,None]==0)&(partition[corpus.children]==0))
            edge_bank=np.column_stack([parent,corpus.children[parent,slot]])
        safe=~np.isin(keys(edge_bank[:,0],edge_bank[:,1],corpus.size),data['reserved_relations'])
        edge_bank=edge_bank[safe]
        if not len(edge_bank):raise ValueError('No non-leaking direct physical edges')
        atomic_json(out/'action_edges.json',dict(count=len(edge_bank),fraction=fraction,label=1,
            scope='Real directed physical edges on training scenes/endpoints; protected goals and relation closures excluded.'))
    validation=[k[6:] for k in data if k.startswith('tasks_') and (k=='tasks_validation' or k.endswith('_validation'))]
    start_step,best,best_step,prior_seconds=0,float('inf'),0,0.
    probe_batches=[];loss_window=np.zeros(3,dtype=np.float64)
    token_replay=np.empty((0,2),dtype=np.int64)
    branch_replay=None
    if options.get('token_branch_replay'):
        from .token_branches import BranchReplay
        import shutil
        branch_options=options['token_branch_replay']
        if config['task']!='token_swapping' or not options.get('online_goals') or options.get('token_cycle_replay'):
            raise ValueError('Frozen token branch replay requires its own online-goal condition')
        if not 0<branch_options['decision_fraction']<1 or branch_options['point_count']<1:
            raise ValueError('Positive bounded branch replay quotas required')
        snapshot=out/'token_branch_bank.npz'
        if not snapshot.exists():
            temporary=snapshot.with_suffix('.tmp');shutil.copyfile((out/branch_options['bank']).resolve(),temporary);temporary.replace(snapshot)
        with np.load(snapshot) as archive:bank={k:archive[k] for k in archive.files}
        branch_replay=BranchReplay(corpus,data,np.load(out/'data/endpoint_partition.npy'),bank,branch_options,
            np.random.default_rng(config['seed']+87003))
    continuation=config.get('continuation');external=False
    checkpoint=(out/'last.pt' if resume and (out/'last.pt').exists() else None)
    if checkpoint is None and continuation:
        checkpoint=Path(continuation['checkpoint']).resolve();external=True
    if checkpoint is not None:
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if external:validate_continuation(saved['config'],config,saved['step'])
        elif saved['config']!=config:raise ValueError('Checkpoint scope mismatch')
        if saved['input_dim']!=corpus.input_dim:raise ValueError('Checkpoint input encoding mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng']; pair_sampler.load_state_dict(saved['pair_sampler']); rank_sampler.load_state_dict(saved['rank_sampler'])
        if relation_sampler: relation_sampler.load_coverage_state_dict(saved.get('relation_coverage'))
        probe_batches=saved.get('training_probe_batches',[]);loss_window=saved.get('loss_window',loss_window)
        token_replay=saved.get('token_cycle_replay',token_replay)
        if branch_replay:branch_replay.load_state_dict(saved['token_branch_state'])
        if neighborhood_fraction: neighborhood_rng.bit_generator.state=saved['neighborhood_rng']
        if replay_sampler:
            if not np.array_equal(saved['replay_tasks'],replay_sampler.tasks) or not np.array_equal(saved['replay_reasons'],replay_sampler.reason):
                raise ValueError('Replay bank identity changed during resume')
            replay_rng.bit_generator.state=saved['replay_rng'];replay_draws=saved['replay_draws']
            if isinstance(replay_sampler,LocalPathSampler):replay_sampler.load_state_dict(saved['replay_sampler_state'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        exposure,pair_draws,rank_draws=saved['exposure'],saved['pair_draws'],saved['rank_draws']
        start_step,best,best_step,prior_seconds=saved['step'],saved['best_score'],saved['best_step'],saved['seconds']
        if external:
            import shutil
            for name in ('initial.json','initial_timing.json','best_eligible.json'):
                source=checkpoint.parent/name
                if source.exists() and not (out/name).exists():shutil.copy2(source,out/name)
            if (checkpoint.parent/'best.pt').exists() and not (out/'best.pt').exists():
                retain_checkpoint(checkpoint.parent/'best.pt',out/'best.pt')
            atomic_json(out/'continuation.json',dict(checkpoint=str(checkpoint),source_commit=saved['source_commit'],
                source_step=start_step,original_budget=saved['config']['training']['steps'],extended_budget=options['steps'],
                optimizer_restored=True,sampling_rng_restored=True,sampler_state_restored=True,exposure_restored=True,
                meaning='Full optimizer/RNG/counter continuation with unchanged data, sampling and optimization; only update budget extends.'))
    else:
        initial_checkpoint=options.get('initialize_from')
        if initial_checkpoint:
            initial_path=(out/initial_checkpoint).resolve()
            saved=torch.load(initial_path,map_location=device,weights_only=False)
            if (saved['task']!=config['task'] or saved['scope']!=scope(config,corpus) or saved['config']['model']!=config['model']
                    or saved['config'].get('input_encoding','identity')!=config.get('input_encoding','identity')):
                raise ValueError('Warm-start model scope mismatch')
            model.load_state_dict(saved['model'])
            if not structure_options:
                prior=saved['exposure']
                if len(prior)==corpus.size:exposure=prior.copy()
                elif (config['task'] in ('bridge','game24') and len(prior)<corpus.size
                      and config['data'].get('additional_training_scenes',0)>saved['config']['data'].get('additional_training_scenes',0)
                      and {k:v for k,v in config['data'].items() if k!='additional_training_scenes'}
                          =={k:v for k,v in saved['config']['data'].items() if k!='additional_training_scenes'}):
                    exposure[:len(prior)]=prior
                else:raise ValueError('Warm-start exposure IDs do not match the prepared corpus')
            atomic_json(out/'initialization.json',dict(checkpoint=str(initial_path),source_commit=saved['source_commit'],
                step=saved['step'],optimizer_restored=False,sampler_restored=False,
                prior_exposure_transferred=not bool(structure_options),
                meaning=('Weights warm start; fresh optimizer and sampling phase, not an exact continuation. '
                    + ('Small corpus has remapped state IDs; prior endpoint counters are not transferable.' if structure_options
                       else 'Prior gradient endpoint exposure retained separately from fresh source-row coverage.'))))
    if not (out/'initial.json').exists():
        initial_started=time.monotonic()
        initial=evaluate(model,corpus,data,config,cohorts=validation if options.get('validation_only_initial') else None)
        for name,row in initial['cohorts'].items():
            if row['planning']['oracle']['count'] and row['planning']['oracle']['optimal_rate']!=1:
                raise ValueError(('Oracle planning gate failed',name,row['planning']['oracle']))
        atomic_json(out/'initial.json',initial)
        atomic_json(out/'initial_timing.json',dict(seconds=time.monotonic()-initial_started))
    started=time.monotonic()
    feature_cache=None; feature_ids=None; feature_started=time.monotonic()
    packed_features=options.get('features_on_device') in ('packed_bits','prepared_packed_bits')
    if options.get('features_on_device')=='prepared_packed_bits':
        if not solution_sampler or options.get('online_goals'):
            raise ValueError('Prepared feature cache requires a fixed exact relation bank')
        feature_ids=torch.as_tensor(np.load(out/'data/feature_ids.npy'),device=device)
        packed=np.load(out/'data/features_packed.npy',mmap_mode='r')
        if packed.shape!=(len(feature_ids),(corpus.input_dim+7)//8):raise ValueError('Prepared feature shape mismatch')
        feature_cache=torch.as_tensor(np.array(packed),dtype=torch.uint8,device=device)
    elif options.get('features_on_device'):
        width=(corpus.input_dim+7)//8 if packed_features else corpus.input_dim
        feature_cache=torch.empty((corpus.size,width),dtype=torch.float32 if options.get('features_on_device')=='float32' else torch.uint8,device=device)
        for start in range(0,corpus.size,65536):
            stop=min(start+65536,corpus.size); values=corpus.features(np.arange(start,stop))
            if options.get('features_on_device')!='float32' and not ((values==0)|(values==1)).all(): raise ValueError('Lossless bit-feature cache requires binary input')
            values=np.packbits(values.astype(np.uint8),axis=1,bitorder='little') if packed_features else values
            feature_cache[start:stop]=torch.as_tensor(values,dtype=feature_cache.dtype,device=device)
    feature_seconds=time.monotonic()-feature_started
    convergence=None
    if config.get('convergence'):
        from .solution_convergence import ConvergenceProbe
        convergence=ConvergenceProbe(corpus,data,config,out)
        if not (out/'convergence'/f'{start_step}.json').exists():
            baseline=evaluate(model,corpus,data,config,cohorts=validation,exposure=exposure,planning=True)
            convergence.inspect(model,start_step,pair_draws,rank_draws,baseline,checkpoint)
            if checkpoint is not None:convergence.retain_policy_bests(checkpoint,start_step,baseline)
    trace=[]; trace_number=0; trace_path=None
    if options.get('record_training_samples'):
        parent=out/'samples'; parent.mkdir(exist_ok=True)
        attempt=0
        while (parent/f'attempt_{attempt:03d}').exists(): attempt+=1
        trace_path=parent/f'attempt_{attempt:03d}'; trace_path.mkdir()
    def flush_trace():
        nonlocal trace_number
        if not trace: return
        packed={key:np.concatenate([row[key] for row in trace]) for key in trace[0]}
        target=trace_path/f'{trace_number:05d}.npz'; temporary=target.with_suffix('.tmp')
        with temporary.open('wb') as stream: np.savez_compressed(stream,**packed)
        temporary.replace(target)
        trace.clear(); trace_number+=1
    def save(step,path):
        flush_trace()
        atomic_save(dict(model_type='exact_state_pairs',task=config['task'],scope=scope(config,corpus),
            source_commit=source_commit,config=config,input_dim=corpus.input_dim,step=step,
            model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            pair_sampler=pair_sampler.state_dict(),rank_sampler=rank_sampler.state_dict(),neighborhood_rng=neighborhood_rng.bit_generator.state,
            exposure=exposure,pair_draws=pair_draws,rank_draws=rank_draws,
            replay_rng=replay_rng.bit_generator.state,replay_draws=replay_draws,
            replay_tasks=replay_sampler.tasks if replay_sampler else None,replay_reasons=replay_sampler.reason if replay_sampler else None,
            replay_sampler_state=replay_sampler.state_dict() if isinstance(replay_sampler,LocalPathSampler) else None,
            relation_coverage=relation_sampler.coverage_state_dict() if relation_sampler else None,
            training_probe_batches=probe_batches,loss_window=loss_window,
            token_cycle_replay=token_replay,
            token_branch_state=branch_replay.state_dict() if branch_replay else None,
            best_score=best,best_step=best_step,
            seconds=prior_seconds+time.monotonic()-started),path)
    step=start_step
    for step in range(start_step+1,options['steps']+1):
        model.train()
        epoch=relation_sampler.ranking_epochs if relation_sampler and relation_sampler.coverage_sampler is not None else 0.
        rate=learning_rate_at_epoch(options,epoch)
        for group in optimizer.param_groups: group['lr']=rate
        if solution_sampler:
            p,r=solution_sampler.draw(options['batch_size'],options['decision_batch_size'])
            pair=data['pairs_train'][p];tasks=data['tasks_train'][r];children=data['children_train'][r]
        elif relation_sampler:
            p,r,pair,tasks=relation_sampler.draw(pair_sampler,options['batch_size'],options['decision_batch_size'],rng)
            children=corpus.successors(tasks[:,0])
        else:
            p=pair_sampler.draw(options['batch_size']); r=rank_sampler.draw(options['decision_batch_size'])
            pair=data['pairs_train'][p]; tasks=data['tasks_train'][r]
            children=data['children_train'][r]
        pair,p,point_origin=decision_neighborhood_points(pair,p,tasks,children,
            relation_sampler.point_index if neighborhood_fraction else None,neighborhood_fraction,neighborhood_rng)
        rank_origin=np.zeros(len(tasks),dtype=np.uint8);replay_ids=np.full(len(tasks),-1,dtype=np.int64)
        if replay_sampler:
            pair,p,tasks,r,point_origin,rank_origin,replay_ids=replay_sampler.apply(pair,p,tasks,r,
                relation_sampler.point_index,relation_sampler.rank_index,replay_rng)
            children=corpus.successors(tasks[:,0])
        valid=children>=0
        costs=edge_costs(corpus,tasks[:,0])
        point_target=point_labels[p]
        child_truth=data['child_distances_train'][r]
        if options.get('online_goals') or relation_sampler:
            extra=dict(local_fraction=options.get('local_goal_fraction',0.),walk_max=options.get('goal_walk_max'),
                       distance_bins=options.get('goal_distance_bins'),proposal=options.get('goal_proposal','walk'))
            if not relation_sampler:
                pair=sample_goals(corpus,pair[:,0],rng,data['reserved_relations'],allowed=allowed,fallback=pair[:,1],**extra)
                tasks=sample_goals(corpus,tasks[:,0],rng,data['reserved_relations'],ranking=True,allowed=allowed,fallback=tasks[:,1],**extra)
                if len(token_replay):
                    count=round(len(tasks)*options['token_cycle_replay']['fraction'])
                    slots=rng.choice(len(tasks),count,replace=False)
                    tasks=tasks.copy();r=r.copy()
                    tasks[slots]=token_replay[rng.integers(len(token_replay),size=count)];r[slots]=-1
                    rank_origin[slots]=5
                    children=corpus.successors(tasks[:,0]);valid=children>=0
                    costs=edge_costs(corpus,tasks[:,0])
            point_true=corpus.distance(pair[:,0],pair[:,1])
            point_target=np.where(np.isfinite(point_true),point_true,options['unreachable_target'])
            child_truth=np.full(children.shape,np.inf,dtype=np.float32)
            child_truth[valid]=corpus.distance(children[valid],np.broadcast_to(tasks[:,1,None],children.shape)[valid])
        else: point_true=point_truth[p]
        if branch_replay:
            pair,p,point_true,point_target,tasks,r,point_origin,rank_origin=branch_replay.apply(
                pair,p,point_true,point_target,tasks,r,point_origin,rank_origin)
            children=corpus.successors(tasks[:,0]);valid=children>=0;costs=edge_costs(corpus,tasks[:,0])
            child_truth=corpus.distance(children,np.broadcast_to(tasks[:,1,None],children.shape))
        if edge_bank is not None:
            nedge=round(len(pair)*options['action_edge_fraction'])
            slots=rng.choice(len(pair),nedge,replace=False)
            pair=pair.copy();p=p.copy();point_true=point_true.copy();point_target=point_target.copy()
            pair[slots]=edge_bank[rng.integers(len(edge_bank),size=nedge)]
            p[slots]=-1;point_origin[slots]=4;point_true[slots]=1;point_target[slots]=1
        if len(probe_batches)<options.get('training_probe_batches',0):
            probe_batches.append(dict(pairs=pair.copy(),point_truth=point_true.copy(),tasks=tasks.copy(),
                children=children.copy(),child_truth=child_truth.copy()))
        if trace_path is not None:
            parent_true=corpus.distance(tasks[:,0],tasks[:,1])
            row=dict(point_step=np.full(len(pair),step,dtype=np.int32),pairs=pair.copy(),point_distance=point_true.copy(),
                rank_step=np.full(len(tasks),step,dtype=np.int32),tasks=tasks.copy(),children=children.copy(),
                parent_distance=parent_true,child_distance=child_truth.copy(),
                point_reachability=np.isfinite(point_true),point_origin=point_origin,rank_origin=rank_origin,rank_replay_index=replay_ids,
                optimal=valid & (child_truth+costs==parent_true[:,None]),edge_costs=costs)
            if isinstance(replay_sampler,LocalPathSampler):row.update({key:value.copy() for key,value in replay_sampler.last_annotations.items()})
            trace.append(row)
            if len(trace)>=100: flush_trace()
        ids=np.concatenate([pair[:,0],pair[:,1],children[valid],tasks[:,1]])
        unique,inverse=np.unique(ids,return_inverse=True)
        cache_index=torch.as_tensor(unique,device=device)
        if feature_ids is not None:
            cache_index=torch.searchsorted(feature_ids,cache_index)
            if bool((cache_index>=len(feature_ids)).any()) or not torch.equal(feature_ids[cache_index],torch.as_tensor(unique,device=device)):
                raise ValueError('Training endpoint missing from exact prepared feature cache')
        features=(feature_cache[cache_index] if feature_cache is not None
                  else torch.as_tensor(corpus.features(unique),device=device))
        if packed_features:
            shifts=torch.arange(8,device=device,dtype=torch.uint8)
            features=((features[:,:,None]>>shifts)&1).flatten(1)[:,:corpus.input_dim]
        features=features.float()
        vectors=model.encode(features)
        b,n=len(pair),len(tasks); nchild=int(valid.sum())
        qa,qg=vectors[inverse[:b]],vectors[inverse[b:2*b]]
        pointwise=F.smooth_l1_loss(model.distance(qa,qg),torch.as_tensor(point_target,device=device))
        child=vectors[inverse[2*b:2*b+nchild]]; goal=vectors[inverse[2*b+nchild:]]
        mask=torch.as_tensor(valid,device=device)
        scores=torch.zeros(children.shape,device=device)
        goals=goal[:,None].expand(-1,children.shape[1],-1)[mask]
        scores[mask]=torch.as_tensor(costs,device=device)[mask]+model.distance(child,goals)
        truth=np.where(np.isfinite(child_truth),child_truth,options['unreachable_target'])
        ranking=soft_rank_loss(scores,torch.as_tensor(costs+truth,device=device),mask,options['rank_temperature'])
        loss=pointwise+options['rank_weight']*ranking
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        loss_window+=np.array([float(pointwise.detach()),float(ranking.detach()),1.])
        np.add.at(exposure,ids,1); np.add.at(pair_draws,p[p>=0],1); np.add.at(rank_draws,r[r>=0],1)
        if replay_sampler: np.add.at(replay_draws,replay_ids[replay_ids>=0],1)
        if step==1 or step%options['eval_every']==0 or step==options['steps']:
            planning=bool(convergence or (options.get('rollout_eval_every') and (step%options['rollout_eval_every']==0 or step==options['steps'])))
            values=evaluate(model,corpus,data,config,cohorts=validation,exposure=exposure,planning=planning)
            if options.get('token_cycle_replay') and step%options['eval_every']==0:
                if config['task']!='token_swapping':raise ValueError('Permutation-cycle replay requires token swapping')
                from .relation_evaluate import QCache,rollout
                from .relation_data import keys
                settings=options['token_cycle_replay'];candidates=data['tasks_train']
                queries=candidates[rng.choice(len(candidates),min(len(candidates),settings['queries']),replace=False)]
                records=[];rollout(corpus,QCache(model,corpus,device),queries,settings['horizon'],'q',records)
                repeated=set();previous={};mined=[]
                for edge in records:
                    for j,q in enumerate(edge['query_index']):
                        row=(int(edge['source'][j]),int(edge['goal'][j]))
                        target=int(edge['children'][j,edge['choice'][j]])
                        if int(q) in previous and target==previous[int(q)][0]:
                            repeated.add(int(q));mined.extend([previous[int(q)],row])
                        previous[int(q)]=row
                rows=np.unique(np.concatenate([token_replay,np.asarray(mined,dtype=np.int64).reshape(-1,2)]),axis=0)
                if len(rows):
                    partition=np.load(out/'data'/'endpoint_partition.npy');kids=corpus.successors(rows[:,0])
                    safe=(partition[rows]==0).all(axis=1)&(partition[kids]==0).all(axis=1)
                    safe&=~np.isin(keys(rows[:,0],rows[:,1],corpus.size),data['reserved_relations'])
                    safe&=~np.isin(keys(kids,rows[:,1,None],corpus.size),data['reserved_relations']).any(axis=1)
                    token_replay=rows[safe]
                atomic_json(out/'token_cycle_replay.json',dict(step=step,relations=len(token_replay),
                    queries=len(queries),cycling_queries=len(repeated),scope='Full candidate decisions from training-only cyclic trajectories; all endpoint and relationship exclusions retained.'))
            objective=float(np.mean([row['finite_huber']+row['candidate_soft_rank_loss'] for row in values['cohorts'].values()]))
            covered=bool((pair_draws>0).all())
            relation_options=config.get('relations',{})
            require_coverage=relation_options.get('complete_training_sources',False) or bool(relation_options.get('training_sources_per_scene'))
            eligible=covered or not require_coverage
            # Before full coverage the recovery checkpoint is provisional only.
            previous_eligible=(out/'best_eligible.json').exists()
            improved=(eligible and (not previous_eligible or objective<best)) or (not previous_eligible and objective<best)
            if improved: best,best_step=objective,step
            record=dict(step=step,seconds=round(prior_seconds+time.monotonic()-started,3),
                        distance_loss=float(pointwise.detach()),rank_loss=float(ranking.detach()),
                        validation_score=objective,best_step=best_step,full_source_coverage=covered,
                        source_rows_seen=int((pair_draws>0).sum()),source_rows_total=len(pair_draws),
                        learning_rate=rate,ranking_coverage_epochs=relation_sampler.ranking_epochs if relation_sampler else None,
                        mean_distance_loss=float(loss_window[0]/loss_window[2]),mean_rank_loss=float(loss_window[1]/loss_window[2]),
                        loss_window_batches=int(loss_window[2]),training_probe=training_probe(model,corpus,probe_batches,options),
                        validation={name:dict(mae=row['finite_mae'],shortest_action_rate=row['shortest_action_rate'],
                            **(dict(success_rate=row['planning']['q']['success_rate']) if planning else {})) for name,row in values['cohorts'].items()})
            if branch_replay:
                fitting=evaluate(model,corpus,branch_replay.fit,config,planning=True)
                record['branch_fit']={name:dict(mae=row['finite_mae'],shortest_action_rate=row['shortest_action_rate'],
                    **{k:row['planning']['q'][k] for k in ('count','solved','success_rate','cycles')}) for name,row in fitting['cohorts'].items()}
                gate=options['token_branch_replay'].get('fit_gate',{})
                if gate:
                    record['branch_fit_passed']=all(row['mae']<=gate['maximum_mae'] and row['shortest_action_rate']>=gate['minimum_shortest_action_rate']
                        and row['success_rate']>=gate['minimum_completion_rate'] and row['cycles']/max(row['count'],1)<=gate['maximum_cycle_rate']
                        for row in record['branch_fit'].values())
                atomic_json(out/'branch_fit_latest.json',dict(step=step,metrics=fitting,passed=record.get('branch_fit_passed'),
                    scope='Training-only seed and branch fitting diagnostics; never checkpoint selection or independent validation.'))
            loss_window[:]=0
            with (out/'progress.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            atomic_json(out/'validation_latest.json',values)
            if improved:
                save(step,out/'best.pt')
                if eligible: atomic_json(out/'best_eligible.json',dict(step=step))
            save(step,out/'last.pt')
            if convergence:
                convergence.retain_policy_bests(out/'last.pt',step,values)
                if step in config['convergence']['checkpoints']:
                    retain_checkpoint(out/'last.pt',out/f'checkpoint-{step}.pt')
                    convergence.inspect(model,step,pair_draws,rank_draws,values,out/f'checkpoint-{step}.pt')
            print(json.dumps(record),flush=True)
    final_checkpoint=options.get('evaluation_checkpoint','best.pt')
    if final_checkpoint not in ('best.pt','last.pt'): raise ValueError('Unknown final evaluation checkpoint')
    saved=torch.load(out/final_checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(saved['model'])
    final_started=time.monotonic()
    final=evaluate(model,corpus,data,config,exposure=saved['exposure'])
    summary=dict(task=config['task'],source_commit=source_commit,best_step=best_step,evaluated_step=saved['step'],
                 evaluated_checkpoint=final_checkpoint,last_step=step,
                 preparation_seconds=preparation_seconds,training_seconds=prior_seconds+time.monotonic()-started,
                 initial_evaluation_seconds=json.loads((out/'initial_timing.json').read_text())['seconds'] if (out/'initial_timing.json').exists() else None,
                 final_evaluation_seconds=time.monotonic()-final_started,
                 raw_feature_cache_seconds=feature_seconds,
                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else 0,
                 peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else 0,
                 gpu=torch.cuda.get_device_name(device) if torch.device(device).type=='cuda' else None,
                 data=description,model_parameters=sum(p.numel() for p in model.parameters()),
                 actual_exposure=dict(unique_gradient_states=int((exposure>0).sum()),available_states=corpus.size,
                                      unique_pair_rows=int((pair_draws>0).sum()),pair_rows=len(pair_draws),
                                      unique_decision_rows=int((rank_draws>0).sum()),decision_rows=len(rank_draws),
                                      minimum_pair_layer_passes=pair_sampler.passes,
                                      minimum_decision_layer_passes=None if relation_sampler else rank_sampler.passes),
                 metrics=final)
    if relation_sampler and relation_sampler.coverage_sampler is not None:
        summary['actual_exposure']['ranking_bank_epochs']=relation_sampler.ranking_epochs
        summary['actual_exposure']['ranking_bank_relations']=len(relation_sampler.ranks)
        summary['actual_exposure']['epoch_scope']='Prepared finite training (state, goal) ranking bank; not all physical states or all Boxoban layouts.'
    if replay_sampler:
        summary['failure_replay']=dict(decisions=len(replay_draws),unique_decisions_drawn=int((replay_draws>0).sum()),
            total_context_draws=int(replay_draws.sum()),snapshot=str(out/'failure_replay_bank.npz'),
            counter_scope='Full training budget; counters for the selected model are retained separately in best.pt.')
        if isinstance(replay_sampler,LocalPathSampler):
            summary['failure_replay']['local_paths']=dict(paths=len(replay_sampler.paths),actual_path_context_draws=int(replay_sampler.actual_path_steps.sum()),
                paths_with_any_context_draw=int((replay_sampler.actual_path_steps>0).sum()),completed_sampling_blocks=replay_sampler.blocks-int(replay_sampler.cursor<replay_sampler.options['block_batches']))
    if config.get('acceptance'):
        criteria=json.loads(Path(config['acceptance']).read_text())
        required=criteria['required_test_cohorts'][config['task']]
        rates={name:final['cohorts'][name]['planning']['q']['success_rate'] for name in required}
        coverage=bool((saved['pair_draws']>0).all())
        summary['acceptance']=dict(minimum=criteria['minimum_completion_rate'],target=criteria['target_completion_rate'],
            rates=rates,full_training_source_coverage=coverage,
            passed=coverage and all(rate is not None and rate>=criteria['minimum_completion_rate'] for rate in rates.values()),
            target_reached=coverage and all(rate is not None and rate>=criteria['target_completion_rate'] for rate in rates.values()))
    if config['task']=='game24':
        exact_exposure={}
        for sid,scene in enumerate(corpus.scenes):
            counts=saved['exposure'][corpus.offsets[sid]:corpus.offsets[sid+1]]
            for state,count in zip(scene.values,counts):
                if count:
                    key=','.join(map(str,state));exact_exposure[key]=exact_exposure.get(key,0)+int(count)
        atomic_json(out/'exact_rational_exposure.json',dict(checkpoint=final_checkpoint,counts=exact_exposure,
            semantics='Loss-input occurrences aggregated by sorted reduced rational multiset across questions; not proven nonzero gradients.'))
        summary['exact_rational_exposure']=dict(unique_states=len(exact_exposure),file='exact_rational_exposure.json')
    if branch_replay:
        state=saved['token_branch_state']
        summary['token_branch_replay']=dict(kind=options['token_branch_replay']['kind'],
            decisions=len(branch_replay.tasks),distance_pairs=len(branch_replay.pairs),
            unique_decisions_drawn=int((state['rank_draws']>0).sum()),decision_draws=int(state['rank_draws'].sum()),
            unique_distance_pairs_drawn=int((state['point_draws']>0).sum()),distance_pair_draws=int(state['point_draws'].sum()),
            scope='Selected checkpoint counters. Supplemental distance examples do not displace the original point-source traversal.')
    atomic_json(out/'summary.json',summary)
    print(json.dumps(dict(complete=str(out),best_step=saved['step'])),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True,type=Path); parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--source-commit',required=True); parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',action='store_true'); parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    train(json.loads(args.config.read_text()),args.out,args.source_commit,args.device,args.resume,args.prepare_only)


if __name__=='__main__': main()
