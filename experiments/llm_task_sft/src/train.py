"""Four-GPU DDP full SFT, epoch checkpoints and measured rolling ETA."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .protocol import atomic_json
from .training import Collator, RouteDataset, SupervisedLM


def scheduler_factor(step, total, warmup, floor):
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2


def save_epoch(base, tokenizer, optimizer, scheduler, out, state, rank):
    directory = out/f"checkpoint-{state['step']}"
    directory.mkdir(parents=True, exist_ok=True)
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state())
    torch.save(rng, directory/f'rng_state_{rank}.pt')
    if rank == 0:
        base.save_pretrained(directory, safe_serialization=True, max_shard_size='4GB')
        tokenizer.save_pretrained(directory)
        torch.save(optimizer.state_dict(), directory/'optimizer.pt')
        torch.save(scheduler.state_dict(), directory/'scheduler.pt')
        atomic_json(directory/'training_state.json', state)
    dist.barrier()
    if rank == 0:
        atomic_json(directory/'evaluation_ready.json', dict(epoch=state['epoch'], step=state['step'],
                                                           checkpoint=str(directory)))
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'data', 'model-path', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--longest-smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.update(config['execution'].get('nccl_environment',{}))
    spec = config['training']
    rank, world, local_rank = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    if world != spec['world_size'] or world * spec['per_device_batch_size'] * spec['gradient_accumulation_steps'] != spec['global_batch_size']:
        raise ValueError('DDP launch differs from the formal batch contract')
    if spec['gradient_accumulation_steps'] != 1:
        raise ValueError('This run implements one update per global batch')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.manual_seed(config['seed'] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    data = RouteDataset(args.data)
    if not args.smoke_steps and (len(data) != config['data']['samples_per_epoch'] or data.metadata['audit']['smoke']):
        raise ValueError('Formal training requires the completed 50,000-row dataset')
    if len(data) % spec['global_batch_size']:
        raise ValueError('Dataset must divide into complete global batches')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    # FP32 master parameters / gradients / Adam states; BF16 forward and backward compute.
    base = AutoModelForCausalLM.from_pretrained(args.resume or args.model_path, torch_dtype=torch.float32,
                                               attn_implementation=spec['attention'], local_files_only=True)
    base.config.use_cache = False
    base.to(local_rank)
    if spec['gradient_checkpointing']:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    wrapped = SupervisedLM(base, spec['loss_chunk_tokens'])
    model = DistributedDataParallel(wrapped, device_ids=[local_rank], find_unused_parameters=False,
                                   gradient_as_bucket_view=True)
    groups = [dict(params=[p for n,p in base.named_parameters() if p.requires_grad and p.ndim >= 2], weight_decay=spec['weight_decay']),
              dict(params=[p for n,p in base.named_parameters() if p.requires_grad and p.ndim < 2], weight_decay=0.)]
    optimizer = torch.optim.AdamW(groups, lr=spec['learning_rate'], fused=True)
    steps_epoch = len(data) // spec['global_batch_size']
    total = steps_epoch * spec['epochs']
    warmup = round(total * spec['warmup_fraction'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: scheduler_factor(s, total, warmup, spec['cosine_minimum_ratio']))
    step, epoch_start = 0, 0
    if args.resume:
        saved = json.loads((args.resume/'training_state.json').read_text())
        if saved['config'] != config or saved['steps_per_epoch'] != steps_epoch:
            raise ValueError('Resume contract differs from saved checkpoint')
        optimizer.load_state_dict(torch.load(args.resume/'optimizer.pt', map_location=f'cuda:{local_rank}', weights_only=False))
        scheduler.load_state_dict(torch.load(args.resume/'scheduler.pt', weights_only=False))
        rng = torch.load(args.resume/f'rng_state_{rank}.pt', weights_only=False)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'])
        step, epoch_start = saved['step'], saved['epoch']
    args.out.mkdir(parents=True, exist_ok=True)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=config['seed'])
    loader = DataLoader(data, batch_size=spec['per_device_batch_size'], sampler=sampler,
                        collate_fn=Collator(tokenizer.pad_token_id), pin_memory=True, num_workers=0)
    durations, start_time = deque(maxlen=100), time.monotonic()
    longest = max(range(len(data)), key=lambda i: data.rows[i]['length'])
    log = (args.out/'train.jsonl').open('a') if rank == 0 else None
    try:
        for epoch in range(epoch_start, spec['epochs']):
            sampler.set_epoch(epoch)
            model.train()
            for batch in loader:
                if args.longest_smoke:
                    batch = Collator(tokenizer.pad_token_id)([data[longest]])
                begin = time.monotonic()
                batch = {k:v.to(local_rank, non_blocking=True) for k,v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=spec['bf16']):
                    loss = model(**batch)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(base.parameters(), spec['gradient_clip_norm'])
                optimizer.step(); scheduler.step()
                torch.cuda.synchronize()
                step += 1
                durations.append(time.monotonic() - begin)
                if step % spec['logging_steps'] == 0 or step == 1:
                    diagnostic = torch.tensor([loss.item(), float(norm)], device=local_rank)
                    dist.all_reduce(diagnostic); diagnostic /= world
                    if rank == 0:
                        seconds_step = sum(durations)/len(durations)
                        state = dict(step=step, epoch=step/steps_epoch, total_steps=total, steps_per_epoch=steps_epoch,
                            loss=diagnostic[0].item(), gradient_norm=diagnostic[1].item(), lr=scheduler.get_last_lr()[0],
                            seconds_per_step=seconds_step, eta_seconds=(total-step)*seconds_step,
                            next_epoch_eta_seconds=(steps_epoch-step%steps_epoch)*seconds_step,
                            elapsed_seconds=time.monotonic()-start_time, utc=datetime.now(timezone.utc).isoformat(),
                            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
                        atomic_json(args.out/'progress.json', state)
                        log.write(json.dumps(state)+'\n'); log.flush()
                        print(json.dumps(state), flush=True)
                if args.smoke_steps and step >= args.smoke_steps:
                    save_epoch(base, tokenizer, optimizer, scheduler, args.out,
                        dict(step=step, epoch=epoch, config=config, steps_per_epoch=steps_epoch, smoke=True), rank)
                    if rank == 0:
                        atomic_json(args.out/'complete.json', dict(smoke=True, steps=step))
                    return
            save_epoch(base, tokenizer, optimizer, scheduler, args.out,
                       dict(step=step, epoch=epoch+1, config=config, steps_per_epoch=steps_epoch, smoke=False), rank)
        if rank == 0:
            atomic_json(args.out/'complete.json', dict(epochs=spec['epochs'], steps=step))
    finally:
        if log: log.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
