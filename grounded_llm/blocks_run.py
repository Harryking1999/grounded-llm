"""Blocks task orchestration; shared harness owns all training primitives."""
from pathlib import Path
import time

from .artifacts import append_json, write_json
from .blocks import (build_blocks_dataset, report_examples, planning_prompt,
                     score_step, strict_json, summarize_blocks)


def episode(interface, adapter, task, case, config):
    mask = task.from_grid(case['grid'])
    enabled = adapter is not None
    initial = planning_prompt(task, case, config)
    ids = interface.chat(initial)
    ids += interface.state_suffix(enabled)
    states = [task.to_rows(mask)] if enabled else []
    used, prefill, steps, failure = 0, 0, [], None
    started = time.perf_counter()
    while mask:
        remaining = config['generation']['max_new_tokens'] - used
        space = config['generation']['context_limit'] - len(ids)
        if min(remaining, space) <= 0:
            failure = 'budget_exhausted' if remaining <= 0 else 'context_cap'
            break
        prefill += len(ids)
        generated, raw = interface.next_step(ids, states, adapter, min(remaining, space))
        used += len(generated)
        scored = score_step(task, mask, raw)
        after = scored.pop('after')
        row = dict(raw_output=raw, generated_ids=generated, generated_tokens=len(generated),
                   before=task.to_rows(mask), after=task.to_rows(after), **scored)
        steps.append(row)
        if not row['accepted']:
            # Exhaustion is separate from a fully emitted wrong action/report.
            failure = row['failure']
            if failure == 'format_error' and len(generated) == min(remaining, space):
                failure = 'budget_exhausted' if remaining <= space else 'context_cap'
            break
        mask = after
        if not mask:
            break
        if interface.end_id in generated:
            failure = 'stopped_early'
            break
        ids += generated + interface.state_suffix(enabled)
        if enabled:
            states.append(task.to_rows(mask))
    return dict(id=case['id'], difficulty=case['difficulty'], condition='text_token' if enabled else 'text',
                solved=mask == 0 and failure is None, failure=failure, steps=steps,
                legal_actions=sum(r['action_legal'] for r in steps),
                accepted_steps=sum(r['accepted'] for r in steps),
                generated_tokens=used, prefill_tokens=prefill,
                seconds=time.perf_counter() - started)


def run_blocks(config):
    import torch
    from .harness import provenance
    from .model import load_model
    from .interface import make_adapter
    from .training import train, load_adapter, optimizer_for, update, mean_loss
    from .blocks_interface import BlocksInterface

    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    task, dataset = build_blocks_dataset(config)
    write_json(output / 'dataset.json', dataset)
    write_json(output / 'provenance.json', provenance(config))
    model, tokenizer = load_model(config)
    interface = BlocksInterface(config, model, tokenizer)
    adapter = make_adapter(config, 'mlp', model.config.hidden_size, interface.device)
    training = report_examples(dataset['train'])
    validation = report_examples(dataset['validation'])

    def validate(active_interface, active_adapter, items, epoch):
        correct = 0
        for item in items:
            raw = active_interface.generate_report(item, active_adapter)
            try:
                matched = strict_json(raw) == item['rows']
            except ValueError:
                matched = False
            correct += matched
            append_json(output / 'report_predictions.jsonl', dict(id=item['id'], epoch=epoch,
                        correct=matched, raw_output=raw, expected=item['rows']))
        print(dict(epoch=epoch, report_correct=correct, total=len(items)), flush=True)
        return correct, len(items)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if config['smoke']['enabled']:
        items = training[:config['smoke']['training_report_examples']]
        optimizer = optimizer_for(config, adapter)
        before = mean_loss(interface, adapter, items, config['training']['initial_microbatch_size'])
        curve = []
        for i in range(config['smoke']['optimizer_steps']):
            loss, grad = update(interface, adapter, optimizer, items, config['training']['initial_microbatch_size'])
            curve.append(dict(step=i + 1, loss=loss, gradient=grad))
            print(curve[-1], flush=True)
        after = mean_loss(interface, adapter, items, config['training']['initial_microbatch_size'])
        batch = interface.batch(items[:2], adapter, supervised=True)
        slot_labels_masked = bool(torch.all(batch['labels'][batch['input_ids'] == interface.slot_id] == -100))
        result = dict(loss_before=before, loss_after=after, curve=curve,
                      frozen_gradients_absent=all(p.grad is None for p in model.parameters()),
                      slot_labels_masked=slot_labels_masked,
                      seconds=time.perf_counter() - started,
                      peak_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(output / 'smoke.json', result)
        if not after < before or not slot_labels_masked:
            raise RuntimeError('Smoke failed; inspect saved evidence')
        validate(interface, adapter, items[:2], 'smoke')
    else:
        directory = output / 'adapter'
        directory.mkdir()
        train(interface, adapter, training, validation, directory, validate)
        load_adapter(directory / 'selected.safetensors', adapter, interface.device)
    write_json(output / 'training_cost.json', dict(seconds=time.perf_counter() - started,
               peak_memory_bytes=torch.cuda.max_memory_allocated(),
               trainable_parameters=sum(p.numel() for p in adapter.parameters())))
    adapter.eval().requires_grad_(False)
    probe_count = config['evaluation']['report_probe_count']
    # Interleave difficulties so the readout probe covers the same difficulty range.
    probes = report_examples(sorted(dataset['test'], key=lambda c: (c['id'].rsplit('_', 1)[-1], c['difficulty'])))[:probe_count]
    correct, total = validate(interface, adapter, probes, 'heldout_selected')
    write_json(output / 'readout.json', dict(correct=correct, total=total))
    rows = []
    for case in dataset['test']:
        for condition in config['conditions']:
            row = episode(interface, adapter if condition['adapter'] else None, task, case, config)
            append_json(output / 'episodes.jsonl', row)
            rows.append(row)
            print(dict(id=row['id'], condition=row['condition'], solved=row['solved'],
                       failure=row['failure'], accepted_steps=row['accepted_steps']), flush=True)
    result = summarize_blocks(rows, config)
    result['readout'] = dict(correct=correct, total=total)
    result['smoke_only'] = config['smoke']['enabled']
    write_json(output / 'summary.json', result)
    return result
