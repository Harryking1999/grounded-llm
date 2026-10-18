"""Two-rank DDP with exact global supervised-token mean and half-epoch saves."""
import argparse
from contextlib import nullcontext
from collections import deque
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from transformers import AutoTokenizer
from experiments.flamingo_map_reader.src.train import collate_examples,to_device
from experiments.llm_task_sft.src.train import scheduler_factor
from .common import atomic_json,pseudo_timeline,renumber_trajectory
from .modeling import build,infer_weights
from .contract import training_steps


def shard_model(model, local):
    from functools import partial
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    return FSDP(model, device_id=local, use_orig_params=True,
                auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                         transformer_layer_cls={Qwen2DecoderLayer}),
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

class Routes(Dataset):
    def __init__(self,path,real_map,*,config=None,baseline=None,tokenizer=None):
        self.path=Path(path);self.meta=json.loads((self.path/'dataset.json').read_text());self.rows=self.meta['rows'];self.key='map' if real_map else 'plain'
        self.config,self.baseline,self.tokenizer,self.epoch=config,baseline,tokenizer,0
        self.renumber=bool(config and config['baselines'][baseline].get('renumber_each_epoch',False))
        if self.renumber and (not real_map or tokenizer is None):raise ValueError('Epoch renumbering requires a map trajectory and tokenizer')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        item=torch.load(self.path/'trajectories'/self.rows[index]['prepared_file'],map_location='cpu',weights_only=False)[self.key]
        if self.renumber:
            from experiments.flamingo_map_reader.src.transcript import encode_trajectory
            demo=renumber_trajectory(self.rows[index],item['steps'],self.config['seed'],self.epoch,index,
                self.config['baselines'][self.baseline].get('response_order','reason_first'))
            encoded=encode_trajectory(demo,self.tokenizer,self.config['maximum_sequence_tokens'])
            return encoded,[turn.step for turn in demo.turns]
        return item['encoded'],item['steps']


def learning_rate_factor(step,total,training):
    total=training.get('scheduler_total_steps',total)
    if total <= 0:raise ValueError('Scheduler horizon must be positive')
    warmup=training.get('warmup_steps',round(total*training['warmup_fraction']))
    schedule=training.get('scheduler','cosine')
    if schedule=='constant_with_warmup':return min(1.,step/max(1,warmup))
    if schedule!='cosine':raise ValueError('Unsupported learning-rate scheduler')
    return scheduler_factor(step,total,warmup,training['cosine_minimum_ratio'])

def validate_resume_contract(saved,config,baseline,allow_horizon_extension=False):
    if allow_horizon_extension:
        previous=saved['config']
        old_training,new_training=previous['training'],config['training']
        old_total,new_total=training_steps(previous),training_steps(config)
        # Only an explicit longer horizon is permitted; the LR prefix must be identical.
        if (new_total<=old_total or
                new_training.get('scheduler_total_steps',0)<new_total or
                old_training.get('scheduler_total_steps')!=new_training.get('scheduler_total_steps')):
            raise ValueError('Extension requires a longer budget and an unchanged explicit scheduler horizon')
        old_recipe={k:v for k,v in old_training.items() if k not in ('epochs','maximum_updates')}
        new_recipe={k:v for k,v in new_training.items() if k not in ('epochs','maximum_updates')}
        if old_recipe!=new_recipe:
            raise ValueError('Extension may only change epochs and maximum_updates')
        from copy import deepcopy
        saved=deepcopy(saved)
        saved['config']['training']=dict(new_training)
    recipe=('model','seed','data','map','training','maximum_sequence_tokens')
    if (saved['baseline']!=baseline or saved['config']['baselines'][baseline]!=config['baselines'][baseline]
            or any(saved['config'][key]!=config[key] for key in recipe)):
        raise ValueError('Resume training recipe mismatch')

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

def main():
    p=argparse.ArgumentParser()
    for name in ('config','data','model-path','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--smoke-steps',type=int);p.add_argument('--resume',type=Path)
    p.add_argument('--extend-training',action='store_true',help='Explicitly extend a completed run without changing its optimizer or LR schedule')
    a=p.parse_args();cfg=json.loads(a.config.read_text());spec=cfg['baselines'][a.baseline];t=dict(cfg['training'],**spec.get('training',{}))
    if a.extend_training and not a.resume:raise ValueError('Extension requires a resume checkpoint')
    rank,local,world=int(os.environ['RANK']),int(os.environ['LOCAL_RANK']),int(os.environ['WORLD_SIZE'])
    accumulation=t['gradient_accumulation_steps']
    if world!=t['world_size'] or world*t['micro_batch']*accumulation!=t['global_batch']:
        raise ValueError('DDP topology differs from the configured global batch')
    torch.cuda.set_device(local);dist.init_process_group('nccl');device=torch.device('cuda',local)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
    data=Routes(a.data,spec['real_map'],config=cfg,baseline=a.baseline,tokenizer=tokenizer)
    if spec['real_map'] and data.meta.get('response_order','reason_first')!=spec.get('response_order','reason_first'):
        raise ValueError('Prepared assistant order differs from baseline configuration')
    ready=json.loads((a.data/'ready.json').read_text())
    if not a.smoke_steps and (len(data)!=cfg['data']['samples_per_epoch'] or ready['smoke']):raise ValueError('Formal training requires accepted complete common data')
    sharded=t.get('distributed_strategy','ddp')=='fsdp'
    model=build(cfg,a.baseline,a.model_path)
    wrapped=shard_model(model,local) if sharded else DDP(model.to(device),device_ids=[local],gradient_as_bucket_view=True)
    params=[p for p in wrapped.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay'],fused=True)
    if len(data)%t['global_batch']:raise ValueError('Dataset must divide into complete global batches')
    steps_epoch=len(data)//t['global_batch'];total=training_steps(dict(cfg,training=t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:learning_rate_factor(s,total,t))
    step=0
    if a.resume:
        state=json.loads((a.resume/'state.json').read_text())
        validate_resume_contract(state,cfg,a.baseline,allow_horizon_extension=a.extend_training)
        saved=torch.load(a.resume/'resume.pt',map_location='cpu' if sharded else device,weights_only=False)
        if sharded:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            with FSDP.state_dict_type(wrapped,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=False)):
                wrapped.load_state_dict(saved['model'])
                optimizer.load_state_dict(FSDP.optim_state_dict_to_load(wrapped,optimizer,saved['optimizer']))
        else:
            model.load_state_dict(saved['model'],strict=False);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);step=state['step']
        rng=torch.load(a.resume/f'rng_{rank}.pt',weights_only=False,map_location='cpu');torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    a.out.mkdir(parents=True,exist_ok=True)
    sampler=DistributedSampler(data,num_replicas=world,rank=rank,seed=cfg['seed'])
    def collate(rows):
        timeline,inputs=collate_examples(rows,tokenizer.pad_token_id or tokenizer.eos_token_id)
        if spec['interface'] and not spec['real_map']:timeline=pseudo_timeline(timeline)
        return timeline,inputs
    loader=DataLoader(data,batch_size=t['micro_batch'],sampler=sampler,num_workers=2,collate_fn=collate)
    windows=deque(maxlen=50);start=time.monotonic();limit=a.smoke_steps or total
    for epoch in range(step//max(1,steps_epoch),t['epochs']):
        data.epoch=epoch;sampler.set_epoch(epoch);it=iter(loader);skip=(step%max(1,steps_epoch))*accumulation
        for _ in range(skip):next(it)
        while step<(epoch+1)*steps_epoch and step<limit:
            micro=[next(it) for _ in range(accumulation)]
            count=torch.tensor(sum(int(inputs['labels'][:,1:].ne(-100).sum()) for _,inputs in micro),device=device,dtype=torch.float64);dist.all_reduce(count)
            optimizer.zero_grad(set_to_none=True);loss_sum=torch.zeros((),device=device);began=time.monotonic()
            for index,(timeline,inputs) in enumerate(micro):
                inputs={k:v.to(device) for k,v in inputs.items()};n=inputs['labels'][:,1:].ne(-100).sum()
                kwargs=dict(inputs)
                if spec['interface']:kwargs['map_batch']=to_device(timeline,device)
                with wrapped.no_sync() if index<accumulation-1 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=wrapped(**kwargs);loss=output.loss if hasattr(output,'loss') else output
                        scaled=loss*n*world/count
                    scaled.backward()
                loss_sum+=loss.detach()*n
            grad=wrapped.clip_grad_norm_(t['gradient_clip_norm']) if sharded else torch.nn.utils.clip_grad_norm_(params,t['gradient_clip_norm'])
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite training gradient')
            optimizer.step();scheduler.step();step+=1;windows.append(time.monotonic()-began);dist.all_reduce(loss_sum)
            if rank==0 and (step%t['log_steps']==0 or step==1):
                log=dict(step=step,epoch=step/steps_epoch,loss=float(loss_sum/count),lr=scheduler.get_last_lr()[0],grad_norm=float(grad),seconds_per_step=sum(windows)/len(windows),eta_seconds=(total-step)*sum(windows)/len(windows),supervised_tokens=int(count),peak_gib=torch.cuda.max_memory_allocated()/2**30)
                with (a.out/'train.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
                atomic_json(a.out/'progress.json',log);print(json.dumps(log),flush=True)
            stop=torch.tensor(int((a.out/'stop_after_checkpoint').exists()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if step%t['save_steps']==0 or step==limit or step==total or bool(stop):
                if sharded:save_sharded(wrapped,optimizer,scheduler,a.out,cfg,a.baseline,step,steps_epoch,rank)
                else:save(model,optimizer,scheduler,a.out,cfg,a.baseline,step,steps_epoch,rank)
                if bool(stop):
                    if rank==0:atomic_json(a.out/'finished.json',dict(status='stopped',step=step))
                    dist.destroy_process_group();return
        if step>=limit:break
    if rank==0:atomic_json(a.out/'finished.json',dict(status='smoke_completed' if a.smoke_steps else 'completed',step=step))
    dist.destroy_process_group()

if __name__=='__main__':main()
