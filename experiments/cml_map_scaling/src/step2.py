"""Execute the committed Step 2 contract: smoke, train both interfaces, then test."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import tokenizers
import transformers
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from .run import ROOT, write_json
from .step2_data import (adjacency_text, examples, geometry_stat, judge, load_assets,
                         make_split, summarize, swap_diagnostics, template_consistency)
from .step2_model import StateInterface, make_adapter


def append_json(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def sync():
    torch.cuda.synchronize()


def optimizer_for(config, adapter):
    train = config['training']
    return torch.optim.AdamW(adapter.parameters(), lr=train['learning_rate'],
                            betas=tuple(train['betas']), eps=train['epsilon'],
                            weight_decay=train['weight_decay'])


def update(interface, adapter, optimizer, items, microbatch):
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for micro in chunks(items, microbatch):
        losses = interface.losses(micro, adapter)
        loss = losses.sum() / len(items)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        total += float(loss.detach())
    grad = torch.nn.utils.clip_grad_norm_(adapter.parameters(), interface.config['training']['gradient_clip_norm'],
                                         error_if_nonfinite=True)
    if not torch.isfinite(grad) or grad <= 0:
        raise FloatingPointError('Adapter gradient must be finite and nonzero')
    if any(p.grad is not None for p in interface.model.parameters()):
        raise RuntimeError('Frozen LM received parameter gradients')
    optimizer.step()
    return total, float(grad)


@torch.no_grad()
def mean_loss(interface, adapter, items, microbatch):
    return sum(float(interface.losses(micro, adapter).sum()) for micro in chunks(items, microbatch)) / len(items)


def predict(interface, adapter, items, batch_size, adj, distances, condition, checkpoint, path=None, extra=None):
    rows = []
    for batch in chunks(items, batch_size):
        outputs = interface.generate(batch, adapter)
        for item, output in zip(batch, outputs):
            row = dict(judge(item, output['raw_output'], adj, distances),
                       condition=condition, checkpoint=checkpoint, **(extra or {}))
            row.update(output)
            rows.append(row)
            if path is not None:
                append_json(path, row)
    return rows


def save_adapter(path, adapter):
    save_file({k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}, str(path))


def smoke(interface, split, adj, distances, output, args):
    config = interface.config
    items = examples(split['train'])[:config['smoke']['training_report_examples']]
    results = {}
    for condition in config['conditions']:
        adapter = make_adapter(config, condition['adapter'], interface.model.config.hidden_size, interface.device)
        optimizer = optimizer_for(config, adapter)
        torch.cuda.reset_peak_memory_stats()
        before = mean_loss(interface, adapter, items, args.microbatch)
        sync()
        started = time.perf_counter()
        curve = []
        for step in range(config['smoke']['optimizer_steps']):
            loss, grad = update(interface, adapter, optimizer, items, args.microbatch)
            row = dict(condition=condition['name'], step=step + 1, report_loss=loss, gradient_norm=grad)
            curve.append(row)
            append_json(output / 'smoke_train.jsonl', row)
            print(json.dumps(row), flush=True)
        sync()
        seconds = time.perf_counter() - started
        after = mean_loss(interface, adapter, items, args.microbatch)
        # A synthetic long ID exercises multiple answer tokens without becoming
        # training data or affecting the freshly initialized formal conditions.
        probe = interface.batch(items[:2], adapter, supervised=True, answer_override=['123456789', '0'])
        counts = (probe['labels'] != -100).sum(dim=1).tolist()
        assert counts[0] > 2 and counts[1] >= 2
        for slot in interface.slot_ids:
            assert torch.all(probe['labels'][probe['input_ids'] == slot] == -100)
        assert torch.all(probe['labels'][probe['attention_mask'] == 0] == -100)
        assert after < before, (condition['name'], before, after)
        generated = predict(interface, adapter, items, args.eval_batch, adj, distances,
                            condition['name'], 'smoke', output / 'smoke_predictions.jsonl')
        results[condition['name']] = dict(
            loss_before=before, loss_after=after, curve=curve,
            frozen_parameter_grad_absent=all(p.grad is None for p in interface.model.parameters()),
            q_gradient_absent=interface.q.grad is None,
            synthetic_answer_token_counts=counts,
            synthetic_supervised_token_ids=[r[r != -100].tolist() for r in probe['labels']],
            slot_and_padding_labels_masked=True,
            seconds_per_step=seconds / config['smoke']['optimizer_steps'],
            samples_per_step=len(items), microbatch=args.microbatch,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            generated_examples=generated[:2])
        write_json(output / 'smoke.json', results)
        del optimizer, adapter, probe
        torch.cuda.empty_cache()
    return results


def train_condition(interface, condition, split, adj, distances, output, args):
    config, name = interface.config, condition['name']
    directory = output / name
    directory.mkdir()
    adapter = make_adapter(config, condition['adapter'], interface.model.config.hidden_size, interface.device)
    optimizer = optimizer_for(config, adapter)
    save_adapter(directory / 'initial.safetensors', adapter)
    training = examples(split['train'])
    validation = examples(split['validation'])
    rng = np.random.default_rng(config['data']['report_shuffle_seed'])
    best, best_epoch = None, None
    total_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config['training']['epochs'] + 1):
        order = rng.permutation(len(training))
        shuffled = [training[int(i)] for i in order]
        weighted_loss = 0.0
        sync()
        started = time.perf_counter()
        for step, batch in enumerate(chunks(shuffled, config['training']['report_batch_size']), 1):
            loss, grad = update(interface, adapter, optimizer, batch, args.microbatch)
            weighted_loss += loss * len(batch)
            append_json(directory / 'train.jsonl', dict(epoch=epoch, step=step, samples=len(batch),
                                                       loss=loss, gradient_norm=grad))
            if step == 1 or step % 10 == 0:
                print(json.dumps(dict(condition=name, epoch=epoch, step=step, loss=loss)), flush=True)
        sync()
        train_seconds = time.perf_counter() - started
        validation_loss = mean_loss(interface, adapter, validation, args.microbatch)
        predictions = predict(interface, adapter, validation, args.eval_batch, adj, distances, name,
                              'validation', directory / 'validation_predictions.jsonl', {'epoch': epoch})
        correct = sum(row['correct'] for row in predictions)
        accuracy = correct / len(predictions)
        rank = (-accuracy, validation_loss, epoch)
        if best is None or rank < best:
            best, best_epoch = rank, epoch
            save_adapter(directory / 'selected.safetensors', adapter)
        row = dict(epoch=epoch, report_train_loss=weighted_loss / len(training),
                   validation_report_loss=validation_loss, validation_correct=correct,
                   validation_total=len(predictions), validation_report_accuracy=accuracy,
                   selected_epoch=best_epoch, train_seconds=train_seconds,
                   epoch_total_seconds=time.perf_counter() - started)
        append_json(directory / 'validation.jsonl', row)
        print(json.dumps(dict(condition=name, **row)), flush=True)
    save_adapter(directory / 'final.safetensors', adapter)
    result = dict(selected_epoch=best_epoch, selected_validation_accuracy=-best[0],
                  selected_validation_loss=best[1], final_epoch=config['training']['epochs'],
                  trainable_parameters=sum(p.numel() for p in adapter.parameters()),
                  total_train_validation_seconds=time.perf_counter() - total_started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(directory / 'training_summary.json', result)
    del optimizer, adapter
    torch.cuda.empty_cache()
    return result


def evaluate(interface, split, q, adj, distances, output, args, training_results):
    config = interface.config
    items = [item for template in config['prompts']['test_templates']
             for item in examples(split['test'], ('report_current', 'report_goal', 'action'), template)]
    summaries = {}
    text = predict(interface, None, items, args.eval_batch, adj, distances,
                   'text_control', 'frozen', output / 'text_control.jsonl')
    summaries['text_control'] = {
        'frozen': {template: summarize([r for r in text if r['template'] == template])
                   for template in config['prompts']['test_templates']}}
    summaries['text_control']['consistency'] = template_consistency(text)
    geometry = {'raw_Q': geometry_stat(q, distances)}
    for condition in config['conditions']:
        name = condition['name']
        directory = output / name
        adapter = make_adapter(config, condition['adapter'], interface.model.config.hidden_size, interface.device)
        condition_summary = {'training': training_results[name]}
        condition_geometry = {'raw_Q': geometry['raw_Q']}
        for checkpoint in config['evaluation']['checkpoints']:
            adapter.load_state_dict(load_file(str(directory / f'{checkpoint}.safetensors'), device=str(interface.device)))
            started = time.perf_counter()
            rows = predict(interface, adapter, items, args.eval_batch, adj, distances,
                           name, checkpoint, directory / 'predictions.jsonl')
            condition_summary[checkpoint] = {
                template: summarize([r for r in rows if r['template'] == template])
                for template in config['prompts']['test_templates']}
            condition_summary[checkpoint]['template_both_correct'] = template_consistency(rows)
            condition_summary[checkpoint]['evaluation_seconds'] = time.perf_counter() - started
            condition_geometry['adapter_' + checkpoint] = geometry_stat(interface.representation(adapter), distances)
            if checkpoint == 'selected':
                swapped = swap_diagnostics(rows, adj, distances)
                write_json(directory / 'swap.json', swapped)
                condition_summary['swap'] = swapped['summary']
            print(json.dumps(dict(condition=name, checkpoint=checkpoint,
                                  summary=condition_summary[checkpoint])), flush=True)
        write_json(directory / 'geometry.json', condition_geometry)
        write_json(directory / 'summary.json', condition_summary)
        summaries[name] = condition_summary
        geometry[name] = condition_geometry
        del adapter
    summary = {'conditions': summaries, 'geometry': geometry,
               'interpretation': 'Single-seed fixed-graph pilot; no unseen-node or cross-graph claim.'}
    write_json(output / 'summary.json', summary)
    write_report(output, summary)


def write_report(output, summary):
    lines = ['# Step 2 结果', '', '只用状态报告训练；动作结果未参与训练或选模。单 seed、固定图试点。', '',
             '| 条件 | 权重 | 模板 | 当前报告 | 目标报告 | 两者正确 | 合法动作 | 一步成功 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    metrics = ('current_report_exact_accuracy', 'goal_report_exact_accuracy', 'both_reports_correct_rate',
               'action_legal_rate', 'one_step_success_rate')
    for name, condition in summary['conditions'].items():
        for checkpoint in ('frozen', 'initial', 'selected', 'final'):
            for template, row in condition.get(checkpoint, {}).items():
                if template not in ('canonical', 'heldout'):
                    continue
                values = [f"{row[m]['successes']}/{row[m]['total']}" for m in metrics]
                lines.append('| ' + ' | '.join([name, checkpoint, template] + values) + ' |')
    lines += ['', '## 表示几何', '', f"原 Q 的距离 Spearman：{summary['geometry']['raw_Q']['spearman']:.6f}", '',
              '| 条件 | 初始 | 选中 | 最终 | 选中轮 | 训练及验证秒数 | 峰值显存 GB |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for name in ('linear_report', 'mlp_report'):
        geo = summary['geometry'][name]
        training = summary['conditions'][name]['training']
        values = [str(geo['adapter_' + ck]['spearman']) for ck in ('initial', 'selected', 'final')]
        lines.append('| ' + ' | '.join([name] + values + [str(training['selected_epoch']),
                     f"{training['total_train_validation_seconds']:.1f}",
                     f"{training['peak_gpu_allocated_bytes'] / 1e9:.2f}"]) + ' |')
    lines += ['', '逐题生成、模板一致性、交换诊断、按距离分组和条件动作成功率见同目录 JSON/JSONL。',
              '运行资产、源码提交、模型来源及依赖版本见 `runtime.json`。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiments/cml_map_scaling/configs/step2.json')
    parser.add_argument('--asset-dir', required=True)
    parser.add_argument('--asset-source-revision')
    parser.add_argument('--asset-source-description')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('smoke', 'run'), required=True)
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--eval-batch', type=int, default=8)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    args.microbatch = args.microbatch or config['training']['initial_microbatch_size']
    assert 1 <= args.microbatch <= config['training']['report_batch_size'] and args.eval_batch > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    q, adj, distances, rms = load_assets(config, args.asset_dir, ROOT)
    split = make_split(config)
    adjacency = adjacency_text(adj)
    write_json(output / 'config.json', config)
    write_json(output / 'split.json', split)
    (output / 'adjacency.txt').write_text(adjacency + '\n', encoding='utf-8')
    runtime = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   asset_source_revision=args.asset_source_revision, asset_source_description=args.asset_source_description,
                   asset_dir=str(Path(args.asset_dir).resolve()), model_dir=str(Path(args.model_dir).resolve()),
                   mode=args.mode, microbatch=args.microbatch, eval_batch=args.eval_batch,
                   python_version=platform.python_version(), numpy_version=np.__version__, torch_version=torch.__version__,
                   transformers_version=transformers.__version__, tokenizers_version=tokenizers.__version__,
                   cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                   q_global_rms=rms, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(output / 'runtime.json', runtime)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True).to('cuda')
    interface = StateInterface(config, model, tokenizer, q, rms, adjacency)
    if args.mode == 'smoke':
        smoke(interface, split, adj, distances, output, args)
    else:
        training_results = {condition['name']: train_condition(interface, condition, split, adj, distances, output, args)
                            for condition in config['conditions']}
        # No test outputs exist until both training/selection procedures finish.
        evaluate(interface, split, q, adj, distances, output, args, training_results)
    runtime['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(output / 'runtime.json', runtime)
    print(json.dumps({'status': 'complete', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
