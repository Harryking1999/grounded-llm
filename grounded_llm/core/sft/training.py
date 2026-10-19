"""Task-independent DDP/FSDP SFT on task-provided prepared trajectories."""
from contextlib import nullcontext
from collections import deque
import json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from grounded_llm.core.files import atomic_json
from grounded_llm.core.reader.batching import collate_examples, to_device, pseudo_timeline
from .modeling import build, infer_weights
from .schedule import training_steps, learning_rate_factor, validate_resume_contract


def shard_model(model, local):
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    # Use the loaded backbone's layer type; wrapped reader layers keep the
    # original decoder layer as a child. This retains the Qwen2 wrap policy.
    base = model.base_model if hasattr(model, 'base_model') else model.base
    layer = base.model.layers[0]
    layer_type = type(layer.decoder_layer) if hasattr(layer, 'decoder_layer') else type(layer)
    return FSDP(model, device_id=local, use_orig_params=True,
                auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                         transformer_layer_cls={layer_type}),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                reduce_dtype=torch.float32,
                                                buffer_dtype=torch.bfloat16))


def save_sharded(model, optimizer, scheduler, out, config, baseline, step, steps_epoch, rank):
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
                                       FullStateDictConfig, FullOptimStateDictConfig)
    path = out / f'checkpoint-{step}'
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                    python=random.getstate(), numpy=np.random.get_state()), path / f'rng_{rank}.pt')
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                             FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        weights = model.state_dict()
        optim = FSDP.optim_state_dict(model, optimizer)
    if rank == 0:
        torch.save(dict(baseline=baseline, weights={k:v.to(torch.bfloat16) if v.is_floating_point() else v
                                                   for k,v in weights.items()}), path / 'weights.pt')
        torch.save(dict(model=weights, optimizer=optim, scheduler=scheduler.state_dict()), path / 'resume.pt')
        atomic_json(path / 'state.json', dict(step=step, epoch=step/steps_epoch, baseline=baseline, config=config))
    dist.barrier()
    if rank == 0:
        atomic_json(path / 'evaluation_ready.json', dict(step=step, epoch=step/steps_epoch, baseline=baseline))
        keep = config['training'].get('keep_resume_checkpoints', 2)
        for old in sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))[:-keep]:
            (old / 'resume.pt').unlink(missing_ok=True)
    dist.barrier()


def save(model,optimizer,scheduler,out,config,baseline,step,steps_epoch,rank):
    path=out/f'checkpoint-{step}';path.mkdir(parents=True,exist_ok=True)
    torch.save(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state()),path/f'rng_{rank}.pt')
    if rank==0:
        torch.save(infer_weights(model,config,baseline),path/'weights.pt')
        # FP32 training state is distinct from the BF16 inference export.
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items() if k in dict(model.named_parameters()) and dict(model.named_parameters())[k].requires_grad},optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),path/'resume.pt')
        atomic_json(path/'state.json',dict(step=step,epoch=step/steps_epoch,baseline=baseline,config=config))
    dist.barrier()
    if rank==0:
        atomic_json(path/'evaluation_ready.json',dict(step=step,epoch=step/steps_epoch,baseline=baseline))
        keep=config['training'].get('keep_resume_checkpoints',2)
        if keep < 1:raise ValueError('At least one resume checkpoint must be retained')
        previous=sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))[:-keep]
        for old in previous:(old/'resume.pt').unlink(missing_ok=True)
    dist.barrier()


def train_update(model, microbatches, optimizer, scheduler, *, device,
                 interface, gradient_clip_norm, sharded=False):
    """One globally token-weighted update, also usable without distributed GPUs."""
    device = torch.device(device)
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1
    count = torch.tensor(sum(int(inputs['labels'][:, 1:].ne(-100).sum())
                             for _, inputs in microbatches), device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(count)
    if count <= 0:
        raise ValueError('An update needs supervised tokens')
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for index, (timeline, inputs) in enumerate(microbatches):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        n = inputs['labels'][:, 1:].ne(-100).sum()
        kwargs = dict(inputs)
        if interface:
            kwargs['map_batch'] = to_device(timeline, device)
        sync = model.no_sync() if distributed and index < len(microbatches) - 1 else nullcontext()
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with sync:
            with autocast:
                output = model(**kwargs)
                loss = output.loss if hasattr(output, 'loss') else output
                scaled = loss * n * world / count
            scaled.backward()
        loss_sum += loss.detach() * n
    params = [p for p in model.parameters() if p.requires_grad]
    grad = (model.clip_grad_norm_(gradient_clip_norm) if sharded
            else torch.nn.utils.clip_grad_norm_(params, gradient_clip_norm))
    if not torch.isfinite(grad):
        raise FloatingPointError('Nonfinite training gradient')
    optimizer.step()
    scheduler.step()
    if distributed:
        dist.all_reduce(loss_sum)
    return loss_sum / count, grad, count


def train(config, baseline, model_path, data, out, *, pad_token_id,
          smoke_steps=None, resume=None, extend_training=False):
    cfg = config
    out = Path(out)
    resume = Path(resume) if resume is not None else None
    spec = cfg['baselines'][baseline]
    t = dict(cfg['training'], **spec.get('training', {}))
    if extend_training and not resume:
        raise ValueError('Extension requires a resume checkpoint')
    rank,local,world=int(os.environ['RANK']),int(os.environ['LOCAL_RANK']),int(os.environ['WORLD_SIZE'])
    accumulation=t['gradient_accumulation_steps']
    if world!=t['world_size'] or world*t['micro_batch']*accumulation!=t['global_batch']:
        raise ValueError('DDP topology differs from the configured global batch')
    torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    sharded=t.get('distributed_strategy','ddp')=='fsdp'
    model=build(cfg,baseline,model_path)
    wrapped=shard_model(model,local) if sharded else DDP(model.to(device),device_ids=[local],gradient_as_bucket_view=True)
    params=[p for p in wrapped.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay'],fused=True)
    if len(data)%t['global_batch']:raise ValueError('Dataset must divide into complete global batches')
    steps_epoch=len(data)//t['global_batch'];total=training_steps(dict(cfg,training=t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:learning_rate_factor(s,total,t))
    step=0
    if resume:
        state=json.loads((resume/'state.json').read_text())
        validate_resume_contract(state,cfg,baseline,allow_horizon_extension=extend_training)
        saved=torch.load(resume/'resume.pt',map_location='cpu' if sharded else device,weights_only=False)
        if sharded:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False)):
                wrapped.load_state_dict(saved['model'])
                optimizer.load_state_dict(FSDP.optim_state_dict_to_load(wrapped,optimizer,saved['optimizer']))
        else:
            model.load_state_dict(saved['model'],strict=False);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);step=state['step']
        rng=torch.load(resume/f'rng_{rank}.pt',weights_only=False,map_location='cpu');torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    out.mkdir(parents=True,exist_ok=True)
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,seed=cfg['seed'])
    def collate(rows):
        timeline,inputs=collate_examples(rows,pad_token_id)
        if spec['interface'] and not spec['real_map']:timeline=pseudo_timeline(timeline)
        return timeline,inputs
    loader=DataLoader(data,batch_size=t['micro_batch'],sampler=sampler,num_workers=2,collate_fn=collate)
    windows=deque(maxlen=50);start=time.monotonic();limit=smoke_steps or total
    for epoch in range(step//max(1,steps_epoch),t['epochs']):
        data.epoch=epoch;sampler.set_epoch(epoch);it=iter(loader);skip=(step%max(1,steps_epoch))*accumulation
        for _ in range(skip):next(it)
        while step<(epoch+1)*steps_epoch and step<limit:
            micro=[next(it) for _ in range(accumulation)]
            began=time.monotonic()
            loss,grad,count=train_update(wrapped,micro,optimizer,scheduler,device=device,
                interface=spec['interface'],gradient_clip_norm=t['gradient_clip_norm'],sharded=sharded)
            step+=1;windows.append(time.monotonic()-began)
            if rank==0 and (step%t['log_steps']==0 or step==1):
                log=dict(step=step,epoch=step/steps_epoch,loss=float(loss),lr=scheduler.get_last_lr()[0],grad_norm=float(grad),seconds_per_step=sum(windows)/len(windows),eta_seconds=(total-step)*sum(windows)/len(windows),supervised_tokens=int(count),peak_gib=torch.cuda.max_memory_allocated()/2**30)
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                atomic_json(out/'progress.json',log);print(json.dumps(log),flush=True)
            stop=torch.tensor(int((out/'stop_after_checkpoint').exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if step%t['save_steps']==0 or step==limit or step==total or bool(stop):
                if sharded:save_sharded(wrapped,optimizer,scheduler,out,cfg,baseline,step,steps_epoch,rank)
                else:save(model,optimizer,scheduler,out,cfg,baseline,step,steps_epoch,rank)
                if bool(stop):
                    if rank==0:atomic_json(out/'finished.json',dict(status='stopped',step=step))
                    dist.destroy_process_group();return
        if step>=limit:break
    if rank==0:atomic_json(out/'finished.json',dict(status='smoke_completed' if smoke_steps else 'completed',step=step))
    dist.destroy_process_group()
