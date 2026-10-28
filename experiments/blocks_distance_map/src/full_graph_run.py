"""Continue the original Q/Adam on exact multi-root graph relations and full forks."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from .full_graph_sampling import JointData
from .failure_sampling import FailureData
from .broad_sampling import BroadData
from .model import BoardEncoder, soft_rank_loss
from .multiboard_run import atomic_save


def relation_batch(batch, device, cap):
    source, goal, truth, rs, rg, rt, groups = batch
    packed = np.concatenate((source, goal, rs, rg))
    states, indices = np.unique(packed, axis=0, return_inverse=True)
    bits = ((states[:, np.arange(100) // 64] >>
             (np.arange(100, dtype=np.uint64) % 64)) & 1).astype(np.float32)
    a, b, c = len(source), len(rs), len(source) * 2
    pairs = np.concatenate((np.stack((indices[:a], indices[a:c]), -1),
                            np.stack((indices[c:c+b], indices[c+b:]), -1)))
    order = np.argsort(groups, kind="stable")
    counts = np.bincount(groups)
    offsets = np.cumsum(counts) - counts
    columns = np.arange(len(groups)) - np.repeat(offsets, counts)
    slots = groups[order] * counts.max() + columns
    return (torch.as_tensor(bits, device=device), torch.as_tensor(pairs, device=device),
            torch.as_tensor(np.where(truth < 0, cap, truth), device=device).float(),
            torch.as_tensor(np.where(rt[order] < 0, cap, rt[order]), device=device).float(),
            torch.as_tensor(order, device=device), torch.as_tensor(slots, device=device),
            (len(counts), int(counts.max())))


def relation_losses(scores, truth, rt, order, slots, shape, temperature):
    point = F.smooth_l1_loss(scores[:len(truth)], truth)
    rank_scores = torch.zeros(shape[0] * shape[1], device=scores.device).scatter(
        0, slots, scores[len(truth):][order]).reshape(shape)
    rank_truth = torch.zeros(shape[0] * shape[1], device=scores.device).scatter(0, slots, rt).reshape(shape)
    valid = torch.zeros(shape[0] * shape[1], device=scores.device, dtype=torch.bool)
    valid[slots] = True; valid = valid.reshape(shape)
    ranking = soft_rank_loss(rank_scores, rank_truth, valid, temperature)
    return point, ranking, rank_scores, rank_truth, valid


def scheduled_sampler(step, samplers, originals, schedule):
    cycles, offset = divmod(step - 1, len(schedule))
    stream = schedule[offset]
    previous = cycles * schedule.count(stream) + schedule[:offset].count(stream)
    pool = originals if stream == 'original' else samplers
    return pool[previous % len(pool)], stream


def run(config, out, source_commit, smoke_steps=None, resume=None):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    options = config["training"]
    torch.set_num_threads(options["cpu_threads"])
    torch.manual_seed(options["seed"])
    parent = torch.load(config["parent_checkpoint"], map_location=device, weights_only=False)
    old = parent["config"]["training"]
    if options["original_cap"] != parent["cap"]:
        raise ValueError("Original encoder coordinate scale changed")
    for key in ("temperature", "listwise_weight", "learning_rate"):
        if options[key] != old[key]:
            raise ValueError(f"Original loss setting changed: {key}")
    model_config = parent["config"]["model"]
    raw = BoardEncoder(model_config["state_dim"], model_config["encoder_hidden_dim"],
                       parent["cap"], parent["metric"]).to(device)
    raw.load_state_dict(parent["model"])
    optimizer = torch.optim.Adam(raw.parameters(), lr=options["learning_rate"])
    optimizer.load_state_dict(parent["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = options["learning_rate"]
    model = DistributedDataParallel(raw, device_ids=[local]) if world > 1 else raw
    sampler_type = (BroadData if options.get("expanded_stream") else
                    FailureData if "failure_diagnosis" in options else JointData)
    sampler_world = int(options.get("sampler_shards", world))
    if world > 1 and sampler_world != world:
        raise ValueError("Distributed sampler shards must match the process count")
    shard_ids = list(range(sampler_world)) if world == 1 else [rank]
    samplers = [sampler_type(config["graph_data"], config["legacy_data"], options, shard, sampler_world)
                for shard in shard_ids]
    original = config.get('original_sampling')
    originals, original_ids = [], []
    schedule = config.get('data_schedule', ['repair'])
    if not schedule or set(schedule) - {'original', 'repair'} or ('original' in schedule and original is None):
        raise ValueError('Data schedule requires the declared original and repair streams')
    if original is not None:
        original_config = json.loads(Path(original['config']).read_text())
        original_options = {**original_config['training'], **original.get('overrides', {})}
        original_world = original['sampler_shards']
        if original_world % world:
            raise ValueError('Original sampler shards must divide evenly across physical ranks')
        original_ids = list(range(rank, original_world, world))
        originals = [BroadData(original['graph_data'], original['legacy_data'], original_options,
                              shard, original_world) for shard in original_ids]
    all_samplers = samplers + originals
    sampler = samplers[0]
    out = Path(out)
    first_step = 0
    if resume is not None:
        restored = torch.load(resume, map_location=device, weights_only=False)
        first_step = restored["continuation_step"]
        raw.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        for shard, sampler in zip(shard_ids, samplers):
            sampler.load_state_dict(torch.load(out / f"sampler_{first_step:06d}_rank{shard}.pt",
                                              map_location="cpu", weights_only=False))
        for shard, sampler in zip(original_ids, originals):
            sampler.load_state_dict(torch.load(out / f"original_sampler_{first_step:06d}_rank{shard}.pt",
                                              map_location='cpu', weights_only=False))
    if rank == 0:
        if resume is None:
            out.mkdir(parents=True, exist_ok=False)
            (out / "checkpoints").mkdir()
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (out / "source_commit.txt").write_text(source_commit + "\n")
    if world > 1:
        dist.barrier()
    started = time.monotonic()
    last = smoke_steps or options["steps"]
    probe = None
    stream_probes = {}
    probe_path=out/f'training_probe_rank{rank}.pt'
    if options.get('training_probe') and resume is not None and probe_path.exists():
        probe=torch.load(probe_path,map_location=device,weights_only=False)
    if originals and options.get('training_probe') and resume is not None:
        for name in set(schedule):
            path = out/f'training_probe_{name}_rank{rank}.pt'
            if path.exists():stream_probes[name] = torch.load(path,map_location=device,weights_only=False)
    for step in range(first_step + 1, last + 1):
        # A single idle GPU can alternate both retained distributed data streams.
        # Each update still uses the original Huber+soft-rank on one full batch.
        sampler, stream = scheduled_sampler(step, samplers, originals, schedule)
        stream_options = original_options if stream == 'original' else options
        begin = time.monotonic()
        arrays = sampler.batch(options["batch_size_per_rank"], options["rank_contexts_per_rank"])
        bits, pairs, truth, rt, order, slots, shape = relation_batch(
            arrays, device, options.get("unreachable_label", parent.get("unreachable_label", parent["cap"])))
        prepared = time.monotonic()
        if originals and options.get('training_probe'):
            if stream not in stream_probes:
                stream_probes[stream] = (bits, pairs, truth, rt, order, slots, shape)
                atomic_save(stream_probes[stream],out/f'training_probe_{stream}_rank{rank}.pt')
            probe = stream_probes[stream]
        elif probe is None and options.get('training_probe'):
            probe = (bits, pairs, truth, rt, order, slots, shape)
            atomic_save(probe,probe_path)
        scores = model(bits, pairs)
        point, ranking, rank_scores, rank_truth, valid = relation_losses(
            scores, truth, rt, order, slots, shape, options['temperature'])
        with torch.no_grad():
            mask = valid.reshape(shape)
            predicted = torch.softmax((-rank_scores / options["temperature"]).masked_fill(~mask, -torch.inf), -1)
            desired = torch.softmax((-rank_truth / options["temperature"]).masked_fill(~mask, -torch.inf), -1)
            nonzero = mask & (predicted != desired)
            counters = sampler.ranking_gradient_counts
            counters["contexts_with_nonzero_score_gradient"] += int(nonzero.any(-1).sum())
            counters["candidates_with_nonzero_score_gradient"] += int(nonzero.sum())
            counters["contexts_with_nonuniform_target"] += int((
                rank_truth.masked_fill(~mask, -torch.inf).max(-1).values !=
                rank_truth.masked_fill(~mask, torch.inf).min(-1).values).sum())
        loss = point + options["listwise_weight"] * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if hasattr(sampler, "record_update"):
            rank_nonzero = torch.zeros(len(order), device=device, dtype=torch.bool)
            rank_nonzero[order] = nonzero.reshape(-1)[slots]
            sampler.record_update(arrays, (scores[:len(truth)].detach() != truth).cpu().numpy(),
                                  rank_nonzero.cpu().numpy())
        if step == 1 or step % options["log_every"] == 0 or step == smoke_steps or (originals and step % options['log_every'] == 1):
            if rank == 0:
                record = {"step": step, "seconds": time.monotonic() - started,
                          "sampling_seconds": prepared - begin,
                          "pointwise": point.item(), "soft_rank": ranking.item(),
                          "unique_batch_states": len(bits), "exposure": sampler.report()}
                if originals:record.update(data_stream=stream,sampler_shard=sampler.rank)
                if probe is not None and (step == 1 or step % options['checkpoint_every'] == 0 or step == last
                                          or (originals and step % options['checkpoint_every'] == 1)):
                    with torch.no_grad():
                        pb, pp, pt, prt, po, ps, pshape = probe
                        probe_prediction=raw(pb,pp)
                        ph, pr, predicted, actual, mask = relation_losses(
                            probe_prediction, pt, prt, po, ps, pshape, options['temperature'])
                        target = torch.softmax((-actual/options['temperature']).masked_fill(~mask,-torch.inf),-1)
                        entropy = -(target*target.clamp_min(1e-30).log()).sum(-1).mean()
                        nontrivial = actual.masked_fill(~mask,-torch.inf).amax(-1) != actual.masked_fill(~mask,torch.inf).amin(-1)
                        choice = predicted.masked_fill(~mask,torch.inf).argmin(-1)
                        correct = actual.gather(1,choice[:,None]).ravel() == actual.masked_fill(~mask,torch.inf).amin(-1)
                        record['training_probe'] = dict(huber=float(ph),rank=float(pr),rank_kl=float(pr-entropy),
                            sampler_rank=rank,
                            nontrivial_decisions=int(nontrivial.sum()))
                        legacy_pairs=round(len(pt)*stream_options['legacy_fraction'])
                        legacy_rank=round(options['rank_contexts_per_rank']*stream_options['legacy_fraction'])
                        record['training_probe']['legacy_huber']=float(F.smooth_l1_loss(
                            probe_prediction[:legacy_pairs],pt[:legacy_pairs]))
                        record['training_probe']['root_huber']=float(F.smooth_l1_loss(
                            probe_prediction[legacy_pairs:len(pt)],pt[legacy_pairs:]))
                        for name,begin,end in (('legacy_best_relation_rate',0,legacy_rank),
                                              ('physical_shortest_action_rate',legacy_rank,len(correct))):
                            take=nontrivial[begin:end]
                            record['training_probe'][name]=(float(correct[begin:end][take].float().mean())
                                                           if take.any() else None)
                with (out / "progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
        if step % options["checkpoint_every"] == 0 or step == last:
            # Sampler states stay separate so NumPy RNG metadata never enters
            # inference checkpoints loaded by the existing weights-only reader.
            for shard, local_sampler in zip(shard_ids, samplers):
                atomic_save(local_sampler.state_dict(), out / f"sampler_{step:06d}_rank{shard}.pt")
            for shard, local_sampler in zip(original_ids, originals):
                atomic_save(local_sampler.state_dict(),out/f'original_sampler_{step:06d}_rank{shard}.pt')
            checkpoint_sampler = samplers[0] if originals else sampler
            actions = torch.as_tensor(np.logical_or.reduce([s.action_exposed for s in all_samplers]),
                                      device=device, dtype=torch.int32)
            if world > 1:
                dist.all_reduce(actions, op=dist.ReduceOp.MAX)
            reports = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(checkpoint_sampler.report(), reports, dst=0)
            else:
                reports = [s.report() for s in samplers]
            sketches = None
            original_reports = None
            if originals:
                local_reports = [dict(shard=shard,report=s.report()) for shard,s in zip(original_ids,originals)]
                original_reports = [None]*world if rank == 0 else None
                if world > 1:dist.gather_object(local_reports,original_reports,dst=0)
                else:original_reports = [local_reports]
            if hasattr(checkpoint_sampler, "sketch_state"):
                sketches = [None] * world if rank == 0 else None
                local_sketch = {"sketches": checkpoint_sampler.sketch_state(), "roots": checkpoint_sampler.phase_roots}
                if world > 1:
                    dist.gather_object(local_sketch, sketches, dst=0)
                else:
                    sketches = [{"sketches": s.sketch_state(), "roots": s.phase_roots} for s in samplers]
            if rank == 0:
                path = out / "checkpoints" / f"step_{step:06d}.pt"
                atomic_save({"model_type": "board_mlp", "model": raw.state_dict(),
                             "optimizer": optimizer.state_dict(), "step": parent["step"] + step,
                             "continuation_step": step, "metric": parent["metric"], "cap": parent["cap"],
                             "unreachable_label": options.get("unreachable_label", parent.get("unreachable_label", parent["cap"])),
                             "config": {"model": model_config, "training": options},
                             "source_commit": source_commit}, path)
                exposure = checkpoint_sampler.report()
                exposure["all_rank_action_types_in_candidate_rows"] = int(actions.sum().item())
                exposure["physical_world_size"] = world
                exposure["sampler_shards"] = sampler_world
                exposure["per_rank_draw_reports" if world > 1 or sampler_world == 1
                         else "serialized_sampler_shard_reports"] = reports
                if original_reports is not None:
                    exposure['original_sampling'] = original_reports
                    exposure['data_schedule'] = schedule
                if sketches is not None:
                    exposure["global_stream_phase"] = checkpoint_sampler.merged_exposure([s["sketches"] for s in sketches])
                    exposure["global_stream_phase"]["distinct_roots"] = int(np.logical_or.reduce([s["roots"] for s in sketches]).sum())
                    exposure["global_stream_phase"]["distance_by_goal_draws"] = {}
                    for report in reports:
                        for key, histogram in report["stream_phase"]["local_distance_by_goal"].items():
                            merged = exposure["global_stream_phase"]["distance_by_goal_draws"].setdefault(key, {})
                            for distance, count in histogram.items():
                                merged[distance] = merged.get(distance, 0) + count
                    for key in ("branch_bundle_actual_update_draws", "branch_bundle_actual_distance_draws",
                                "legacy_actual_huber_distances"):
                        merged = exposure["global_stream_phase"][key] = {}
                        for report in reports:
                            for name, value in report["stream_phase"].get(key, {}).items():
                                if isinstance(value, dict):
                                    bucket = merged.setdefault(name, {})
                                    for distance, count in value.items():
                                        bucket[distance] = bucket.get(distance, 0) + count
                                else:
                                    merged[name] = merged.get(name, 0) + value
                (out / f"exposure_{step:06d}.json").write_text(json.dumps(exposure, indent=2) + "\n")
            if world > 1:
                dist.barrier()
    for sampler in all_samplers:
        sampler.close()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.out, args.source_commit, args.smoke_steps, args.resume)


if __name__ == "__main__":
    main()
