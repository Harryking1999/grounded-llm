"""State-pair mode of the existing new_tasks_qmap entry; one task per invocation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.core.files import atomic_json, atomic_save
from grounded_llm.core.qmap.model import StateEncoder, soft_rank_loss
from grounded_llm.core.qmap.sampling import StratifiedEpochSampler, CoverageSampler
from .relation_data import prepare, load, sample_goals
from .relation_evaluate import evaluate


def scope(config,corpus):
    if config['task']=='token_swapping': return corpus.scope
    return dict(board=[10,10] if config['task']=='sokoban' else [6,6],
                boxes=4 if config['task']=='sokoban' else None,wall_count=0 if config['task']=='rush_hour' else None,
                complete_goal=True,shared_across_layouts=True)


def train(config,out,source_commit,device,resume=False,prepare_only=False):
    out=Path(out)
    if out.exists() and not resume:
        raise FileExistsError('Existing exact-pair run; use --resume with its original config')
    out.mkdir(parents=True,exist_ok=True)
    if resume and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Resume requires the unchanged formal configuration')
    atomic_json(out/'config.json',config)
    reference=config.get('data',{}).get('prepared_from')
    if reference and not (out/'data').exists():
        source=(out/reference).resolve()
        previous=json.loads((source.parent/'config.json').read_text())
        if any(previous[key]!=config[key] for key in ('task','seed','relations')):
            raise ValueError('Prepared relation reference changes task, seed or fixed splits')
        if {k:v for k,v in previous['data'].items() if k!='prepared_from'}!={k:v for k,v in config['data'].items() if k!='prepared_from'}:
            raise ValueError('Prepared relation reference changes physical data scope')
        (out/'data').symlink_to(source,target_is_directory=True)
    prepared=time.monotonic()
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
    rng=np.random.default_rng(config['seed']+2)
    model=StateEncoder(input_dim=corpus.input_dim,**config['model']).to(device)
    if torch.device(device).type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=options['learning_rate'])
    point_truth=data['pair_distances_train']
    point_labels=np.where(np.isfinite(point_truth),point_truth,options['unreachable_target']).astype(np.float32)
    def sampler(labels):
        if options.get('broad_fraction'):
            return CoverageSampler(labels,rng,len(labels),options['broad_fraction'])
        return StratifiedEpochSampler(labels,rng)
    pair_sampler=sampler(point_labels); rank_sampler=sampler(data['distances_train'])
    exposure=np.zeros(corpus.size,dtype=np.uint32)
    pair_draws=np.zeros(len(point_truth),dtype=np.uint32); rank_draws=np.zeros(len(data['tasks_train']),dtype=np.uint32)
    allowed=None
    if options.get('online_goals') and not hasattr(corpus,'scenes'):
        allowed=np.flatnonzero(np.load(out/'data'/'endpoint_partition.npy')==0)
    validation=[k[6:] for k in data if k.startswith('tasks_') and (k=='tasks_validation' or k.endswith('_validation'))]
    start_step,best,best_step,prior_seconds=0,float('inf'),0,0.
    if resume and (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if saved['config']!=config or saved['input_dim']!=corpus.input_dim: raise ValueError('Checkpoint scope mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng']; pair_sampler.load_state_dict(saved['pair_sampler']); rank_sampler.load_state_dict(saved['rank_sampler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
        exposure,pair_draws,rank_draws=saved['exposure'],saved['pair_draws'],saved['rank_draws']
        start_step,best,best_step,prior_seconds=saved['step'],saved['best_score'],saved['best_step'],saved['seconds']
    else:
        initial_checkpoint=options.get('initialize_from')
        if initial_checkpoint:
            initial_path=(out/initial_checkpoint).resolve()
            saved=torch.load(initial_path,map_location=device,weights_only=False)
            if saved['task']!=config['task'] or saved['scope']!=scope(config,corpus) or saved['config']['model']!=config['model']:
                raise ValueError('Warm-start model scope mismatch')
            model.load_state_dict(saved['model']); exposure=saved['exposure'].copy()
            atomic_json(out/'initialization.json',dict(checkpoint=str(initial_path),source_commit=saved['source_commit'],
                step=saved['step'],optimizer_restored=False,sampler_restored=False,
                meaning='Weights warm start; fresh optimizer and sampling phase, not an exact continuation. Prior gradient endpoint exposure retained separately from fresh source-row coverage.'))
    if not (out/'initial.json').exists():
        initial_started=time.monotonic()
        initial=evaluate(model,corpus,data,config)
        for name,row in initial['cohorts'].items():
            if row['planning']['oracle']['count'] and row['planning']['oracle']['optimal_rate']!=1:
                raise ValueError(('Oracle planning gate failed',name,row['planning']['oracle']))
        atomic_json(out/'initial.json',initial)
        atomic_json(out/'initial_timing.json',dict(seconds=time.monotonic()-initial_started))
    started=time.monotonic()
    feature_cache=None; feature_started=time.monotonic()
    if options.get('features_on_device'):
        feature_cache=torch.empty((corpus.size,corpus.input_dim),dtype=torch.uint8,device=device)
        for start in range(0,corpus.size,65536):
            stop=min(start+65536,corpus.size); values=corpus.features(np.arange(start,stop))
            if not ((values==0)|(values==1)).all(): raise ValueError('Lossless bit-feature cache requires binary input')
            feature_cache[start:stop]=torch.as_tensor(values,dtype=torch.uint8,device=device)
    feature_seconds=time.monotonic()-feature_started
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
            pair_sampler=pair_sampler.state_dict(),rank_sampler=rank_sampler.state_dict(),
            exposure=exposure,pair_draws=pair_draws,rank_draws=rank_draws,best_score=best,best_step=best_step,
            seconds=prior_seconds+time.monotonic()-started),path)
    step=start_step
    for step in range(start_step+1,options['steps']+1):
        model.train()
        p=pair_sampler.draw(options['batch_size']); r=rank_sampler.draw(options['decision_batch_size'])
        pair=data['pairs_train'][p]; tasks=data['tasks_train'][r]
        children=data['children_train'][r]; valid=children>=0
        point_target=point_labels[p]
        child_truth=data['child_distances_train'][r]
        if options.get('online_goals'):
            extra=dict(local_fraction=options.get('local_goal_fraction',0.),walk_max=options.get('goal_walk_max'))
            pair=sample_goals(corpus,pair[:,0],rng,data['reserved_relations'],allowed=allowed,fallback=pair[:,1],**extra)
            tasks=sample_goals(corpus,tasks[:,0],rng,data['reserved_relations'],ranking=True,allowed=allowed,fallback=tasks[:,1],**extra)
            point_true=corpus.distance(pair[:,0],pair[:,1])
            point_target=np.where(np.isfinite(point_true),point_true,options['unreachable_target'])
            child_truth=np.full(children.shape,np.inf,dtype=np.float32)
            child_truth[valid]=corpus.distance(children[valid],np.broadcast_to(tasks[:,1,None],children.shape)[valid])
        else: point_true=point_truth[p]
        if trace_path is not None:
            parent_true=corpus.distance(tasks[:,0],tasks[:,1])
            trace.append(dict(point_step=np.full(len(pair),step,dtype=np.int32),pairs=pair.copy(),point_distance=point_true.copy(),
                rank_step=np.full(len(tasks),step,dtype=np.int32),tasks=tasks.copy(),children=children.copy(),
                parent_distance=parent_true,child_distance=child_truth.copy(),
                point_reachability=np.isfinite(point_true),optimal=valid & (child_truth+1==parent_true[:,None])))
            if len(trace)>=100: flush_trace()
        ids=np.concatenate([pair[:,0],pair[:,1],children[valid],tasks[:,1]])
        unique,inverse=np.unique(ids,return_inverse=True)
        features=(feature_cache[torch.as_tensor(unique,device=device)].float() if feature_cache is not None
                  else torch.as_tensor(corpus.features(unique),device=device))
        vectors=model.encode(features)
        b,n=len(pair),len(tasks); nchild=int(valid.sum())
        qa,qg=vectors[inverse[:b]],vectors[inverse[b:2*b]]
        pointwise=F.smooth_l1_loss(model.distance(qa,qg),torch.as_tensor(point_target,device=device))
        child=vectors[inverse[2*b:2*b+nchild]]; goal=vectors[inverse[2*b+nchild:]]
        mask=torch.as_tensor(valid,device=device)
        scores=torch.zeros(children.shape,device=device)
        goals=goal[:,None].expand(-1,children.shape[1],-1)[mask]
        scores[mask]=1+model.distance(child,goals)
        truth=np.where(np.isfinite(child_truth),child_truth,options['unreachable_target'])
        ranking=soft_rank_loss(scores,torch.as_tensor(1+truth,device=device),mask,options['rank_temperature'])
        loss=pointwise+options['rank_weight']*ranking
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        np.add.at(exposure,ids,1); np.add.at(pair_draws,p,1); np.add.at(rank_draws,r,1)
        if step==1 or step%options['eval_every']==0 or step==options['steps']:
            values=evaluate(model,corpus,data,config,cohorts=validation,exposure=exposure,planning=False)
            objective=float(np.mean([row['finite_huber']+row['candidate_soft_rank_loss'] for row in values['cohorts'].values()]))
            covered=bool((pair_draws>0).all())
            eligible=covered or not config.get('relations',{}).get('complete_training_sources',False)
            # Before full coverage the recovery checkpoint is provisional only.
            previous_eligible=(out/'best_eligible.json').exists()
            improved=(eligible and (not previous_eligible or objective<best)) or (not previous_eligible and objective<best)
            if improved: best,best_step=objective,step
            record=dict(step=step,seconds=round(prior_seconds+time.monotonic()-started,3),
                        distance_loss=float(pointwise.detach()),rank_loss=float(ranking.detach()),
                        validation_score=objective,best_step=best_step,full_source_coverage=covered,
                        source_rows_seen=int((pair_draws>0).sum()),source_rows_total=len(pair_draws),
                        validation={name:dict(mae=row['finite_mae'],shortest_action_rate=row['shortest_action_rate']) for name,row in values['cohorts'].items()})
            with (out/'progress.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            atomic_json(out/'validation_latest.json',values)
            if improved:
                save(step,out/'best.pt')
                if eligible: atomic_json(out/'best_eligible.json',dict(step=step))
            save(step,out/'last.pt')
            print(json.dumps(record),flush=True)
    saved=torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(saved['model'])
    final_started=time.monotonic()
    final=evaluate(model,corpus,data,config,exposure=saved['exposure'])
    summary=dict(task=config['task'],source_commit=source_commit,best_step=saved['step'],last_step=step,
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
                                      minimum_pair_layer_passes=pair_sampler.passes,minimum_decision_layer_passes=rank_sampler.passes),
                 metrics=final)
    if config.get('acceptance'):
        criteria=json.loads(Path(config['acceptance']).read_text())
        required=criteria['required_test_cohorts'][config['task']]
        rates={name:final['cohorts'][name]['planning']['q']['success_rate'] for name in required}
        coverage=bool((saved['pair_draws']>0).all())
        summary['acceptance']=dict(minimum=criteria['minimum_completion_rate'],target=criteria['target_completion_rate'],
            rates=rates,full_training_source_coverage=coverage,
            passed=coverage and all(rate is not None and rate>=criteria['minimum_completion_rate'] for rate in rates.values()),
            target_reached=coverage and all(rate is not None and rate>=criteria['target_completion_rate'] for rate in rates.values()))
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
