"""Blocks task orchestration; shared harness owns all training primitives."""
from pathlib import Path
import time

from .artifacts import append_json, write_json, read_json, chunks
from .blocks import (build_blocks_dataset, report_examples, planning_prompt,
                     score_step, strict_json, summarize_blocks, require_readable_adapter)


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
    execution = config['execution']
    phase = execution['phase']
    if phase == 'eval' and execution.get('condition') != 'text':
        require_readable_adapter(execution['training_run'], config)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    if phase == 'eval':
        from experiments.sol_dag_blocks.src.tasks import BlocksTask
        source = Path(execution['training_run'])
        source_config = read_json(source / 'config.json')
        for key in ('blocks', 'model', 'adapter', 'generation', 'training'):
            if source_config[key] != config[key]:
                raise ValueError(f'Evaluation disagrees with saved training contract: {key}')
        task = BlocksTask(read_json(config['blocks']['rules_config'])['blocks'])
        dataset = read_json(source / 'dataset.json')
    elif phase == 'fit':
        from experiments.sol_dag_blocks.src.tasks import BlocksTask
        task = BlocksTask(read_json(config['blocks']['rules_config'])['blocks'])
        dataset = read_json(Path(execution['training_run']) / 'dataset.json')
    else:
        task, dataset = build_blocks_dataset(config)
    write_json(output / 'dataset.json', dataset)
    write_json(output / 'provenance.json', provenance(config))
    readout_data = None
    if config.get('readout_data'):
        if phase != 'train':
            raise ValueError('Diverse readout learning is training-only; no automatic planning')
        from .blocks_readout import build_readout_dataset, coverage_summary
        readout_data = build_readout_dataset(task, dataset, config['readout_data'])
        write_json(output / 'readout_dataset.json', readout_data)
        write_json(output / 'coverage.json', coverage_summary(readout_data))
    model, tokenizer = load_model(config)
    interface = BlocksInterface(config, model, tokenizer)
    adapter = make_adapter(config, 'mlp', model.config.hidden_size, interface.device)
    if execution.get('adapter_init'):
        if phase not in ('train', 'fit'):
            raise ValueError('Adapter initialization applies only to training')
        load_adapter(execution['adapter_init'], adapter, interface.device)
    training = report_examples(dataset['train'])
    validation = report_examples(dataset['validation'])
    if readout_data is not None:
        training, validation = readout_data['train'], readout_data['validation']
    if phase == 'fit':
        # Fixed training boards only; neither development nor test labels guide this diagnostic.
        count = config['fit_diagnostic']['examples_per_difficulty']
        selected = [c for difficulty in config['blocks']['construction_counts']
                    for c in [c for c in dataset['train'] if c['difficulty'] == difficulty][:count]]
        training = [dict(id=c['id'], rows=c['grid'].split('/'), task='report_board') for c in selected]
        validation = training
        write_json(output / 'fit_items.json', training)

    def validate(active_interface, active_adapter, items, epoch):
        correct = 0
        records = []
        for batch in chunks(items, config['evaluation']['report_batch_size']):
            for item, raw in zip(batch, active_interface.generate_reports(batch, active_adapter)):
                try:
                    matched = strict_json(raw) == item['rows']
                except ValueError:
                    matched = False
                correct += matched
                row = dict(id=item['id'], epoch=epoch, category=item.get('category', 'constructed'),
                           correct=matched, raw_output=raw, expected=item['rows'])
                records.append(row)
                append_json(output / 'report_predictions.jsonl', row)
        from .blocks import report_diagnostics
        summary = dict(epoch=epoch, overall=report_diagnostics(records), by_category={
            category: report_diagnostics([r for r in records if r['category'] == category])
            for category in sorted({r['category'] for r in records})})
        append_json(output / 'readout_validation.jsonl', summary)
        print(dict(epoch=epoch, report_correct=correct, total=len(items)), flush=True)
        return correct, len(items)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if phase == 'eval':
        if execution['condition'] != 'text':
            load_adapter(source / 'adapter/selected.safetensors', adapter, interface.device)
    elif config['smoke']['enabled']:
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
    if phase != 'eval':
        write_json(output / 'training_cost.json', dict(seconds=time.perf_counter() - started,
                   peak_memory_bytes=torch.cuda.max_memory_allocated(),
                   trainable_parameters=sum(p.numel() for p in adapter.parameters())))
    adapter.eval().requires_grad_(False)
    if readout_data is not None:
        # Keep this training stage entirely separate from the already inspected planning test set.
        selected = read_json(output / 'adapter/training_summary.json')
        result = dict(status='readout_training_complete', training=selected,
                      ready_for_planning=False, planning_not_run=True)
        write_json(output / 'readout_training_summary.json', result)
        return result
    if phase == 'fit':
        correct, total = validate(interface, adapter, training, 'fit_selected')
        # Rotate inputs while retaining original labels: a genuine reader follows the
        # changed state, so original-label accuracy should fall.
        rotated = training[1:] + training[:1]
        outputs = interface.generate_reports(rotated, adapter)
        original_matches, changed_matches = 0, 0
        for original, changed, raw in zip(training, rotated, outputs):
            try:
                parsed = strict_json(raw)
            except ValueError:
                parsed = None
            original_matches += parsed == original['rows']
            changed_matches += parsed == changed['rows']
            append_json(output / 'input_rotation.jsonl', dict(original_id=original['id'],
                        input_id=changed['id'], raw_output=raw, follows_changed_input=parsed == changed['rows']))
        result = dict(status='fit_diagnostic_only', exact=correct, total=total,
                      rotated_original_matches=original_matches, rotated_input_matches=changed_matches,
                      ready_for_planning=False)
        write_json(output / 'fit_summary.json', result)
        return result
    if phase != 'eval' and not config['smoke']['enabled']:
        try:
            require_readable_adapter(output, config)
        except ValueError as error:
            result = dict(status='readout_not_ready', reason=str(error), ready_for_planning=False)
            write_json(output / 'readout.json', result)
            return result
    probe_count = config['evaluation']['report_probe_count']
    # Interleave difficulties so the readout probe covers the same difficulty range.
    probes = report_examples(sorted(dataset['test'], key=lambda c: (c['id'].rsplit('_', 1)[-1], c['difficulty'])))[:probe_count]
    correct, total = None, None
    if phase != 'eval':
        correct, total = validate(interface, adapter, probes, 'heldout_selected')
        write_json(output / 'readout.json', dict(correct=correct, total=total))
    if phase == 'train':
        return dict(status='trained', readout_correct=correct, readout_total=total)
    if phase == 'all':
        if config['smoke']['enabled']:
            return dict(status='smoke_only_no_planning', readout_correct=correct, readout_total=total)
        require_readable_adapter(output, config)
    rows = []
    for case in dataset['test'][execution['shard_index']::execution['num_shards']]:
        for condition in config['conditions']:
            if execution.get('condition') and condition['name'] != execution['condition']:
                continue
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
