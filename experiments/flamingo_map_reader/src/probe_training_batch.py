"""Two real DDP optimizer updates on the longest prepared trajectories."""

from argparse import ArgumentParser
import json
import os
from pathlib import Path
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .train import build_reader, collate_examples, to_device
from .trajectory_dataset import PreparedTrajectoryDataset


def main():
    parser = ArgumentParser()
    for name in ('config', 'manifest', 'model-path', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--stress-profile', type=Path)
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl')
    config = json.loads(args.config.read_text())
    set_seed(config['seed'])
    manifest = json.loads(args.manifest.read_text())
    if args.stress_profile:
        profile = json.loads(args.stress_profile.read_text())
        if Path(profile['manifest']).resolve() != args.manifest.resolve():
            raise ValueError('Stress profile belongs to a different manifest')
        by_id = {r['trajectory_id']: r for r in manifest['records'] if r['split']=='train'}
        selected = profile['selected']
        if args.batch_size < len(selected):
            raise ValueError('Batch must cover every padding extremum')
        examples = [PreparedTrajectoryDataset(args.manifest, [by_id[r['trajectory_id']]], config)[r['variant']]
                    for r in selected]
        examples = [examples[i % len(examples)] for i in range(args.batch_size)]
    else:
        records = sorted((r for r in manifest['records'] if r['split']=='train'),
                         key=lambda r: -r['max_tokens'])[:args.batch_size]
        dataset = PreparedTrajectoryDataset(args.manifest, records, config)
        examples = [dataset[sum(r['variants'] for r in records[:i])] for i in range(len(records))]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    reader = build_reader(base, config).to(rank)
    model = DistributedDataParallel(reader, device_ids=[rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in reader.parameters() if p.requires_grad], lr=2e-5)
    timeline, inputs = collate_examples(examples, tokenizer.pad_token_id)
    timeline = to_device(timeline, rank)
    inputs = {k: v.to(rank) for k,v in inputs.items()}
    losses = []
    started = time.monotonic()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(map_batch=timeline, **inputs)
        output.loss.backward()
        losses.append(output.loss.item())
        if not torch.isfinite(output.loss):
            raise ValueError('Nonfinite probe loss')
        optimizer.step()
        del output
    torch.cuda.synchronize()
    result = dict(rank=rank, per_device_batch=args.batch_size,
        global_batch=args.batch_size*torch.distributed.get_world_size(),
        tokens=inputs['input_ids'].shape[1], losses=losses,
        ffn_gradient_norm=(reader.memory_encoder.feature_ffn[0].weight.grad.norm().item()
            if reader.memory_encoder.feature_ffn is not None else None),
        peak_gib=torch.cuda.max_memory_allocated()/1024**3,
        reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        total_gib=torch.cuda.get_device_properties(rank).total_memory/1024**3,
        snapshots=max(timeline.snapshot_counts), slots=timeline.snapshots.valid.shape[1],
        steps=args.steps, stress_profile=str(args.stress_profile) if args.stress_profile else None,
        seconds=time.monotonic()-started)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/f'rank_{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
