"""Sample official Qwen thinking models through SGLang's native generate API.

The existing Sol module remains authoritative for prompts, judges and summaries.
Runtime model directories and endpoints are command-line inputs, never defaults.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
SOL = ROOT / 'experiments/sol_dag_blocks/src'
sys.path.insert(0, str(SOL))
spec = importlib.util.spec_from_file_location('sol_runner', SOL / 'run.py')
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)


def split_thinking(text):
    # Official template revisions either supply <think> or let the model emit it.
    if '</think>' not in text:
        return text, '', False
    reasoning, answer = text.split('</think>', 1)
    return reasoning.removeprefix('<think>').strip(), answer.strip(), True


def verdict_for(case, text, finish):
    reasoning, answer, closed = split_thinking(text)
    verdict = sol.evaluate(case, answer)
    if finish == 'length':
        verdict = {'pass': False, 'failure_type': 'budget_truncated', 'partial_verdict': verdict}
    elif finish != 'stop':
        raise ValueError(f'Unexpected SGLang finish type: {finish}')
    return reasoning, answer, verdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--model-id', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--tokenizer-path')
    p.add_argument('--endpoints', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    suite = json.loads(Path(args.suite).read_text())
    assert args.model_id in config['models']
    assert len(suite['cases']) == 48
    assert sum(c['replicates'] for c in suite['cases']) == config['max_calls_per_model']
    assert all(c['replicates'] == config['replicates'] for c in suite['cases'])
    from transformers import AutoTokenizer
    import importlib.metadata
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    record_dir = out / 'samples'
    record_dir.mkdir(exist_ok=True)
    metadata = out / 'run_config.json'
    identity = {'config': config, 'model_id': args.model_id, 'model_path': args.model_path,
                'tokenizer_path': tokenizer_path,
                'suite_path': str(Path(args.suite).resolve()),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    if metadata.exists():
        if not args.resume or json.loads(metadata.read_text())['identity'] != identity:
            raise ValueError('Existing run requires --resume and identical contract/source/model/suite')
        if json.loads((out / 'suite.json').read_text()) != suite:
            raise ValueError('Suite contents changed')
    else:
        metadata.write_text(json.dumps({'identity': identity, 'endpoints': args.endpoints,
            'versions': {m: importlib.metadata.version(m) for m in ('sglang', 'torch', 'transformers')},
            'started_at': datetime.now(timezone.utc).isoformat()}, indent=2))
        (out / 'suite.json').write_text(json.dumps(suite, indent=2))
    prompts = {}
    for case in suite['cases']:
        prompt = sol.prompt_for(case)
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=config['enable_thinking'])
        assert rendered.endswith(('<|im_start|>assistant\n', '<think>\n'))
        assert 'enable_thinking' in tokenizer.chat_template
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) + config['sampling']['max_new_tokens'] > config['serving']['context_length']:
            raise ValueError('Context would reduce the requested output budget')
        prompts[case['id']] = {'prompt': prompt, 'rendered_prompt': rendered, 'input_ids': ids}
    (out / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    jobs = [(case, n) for case in suite['cases'] for n in range(1, case['replicates'] + 1)]
    seed_by_slot = {(c['id'], n): config['seed'] + i for i, (c, n) in enumerate(jobs)}

    def sample(case, replicate, endpoint):
        path = record_dir / f"{case['id']}__{replicate}.json"
        if path.exists():
            return json.loads(path.read_text())
        started = time.monotonic()
        seed = seed_by_slot[case['id'], replicate]
        request = {'input_ids': prompts[case['id']]['input_ids'],
                   'sampling_params': dict(config['sampling'], sampling_seed=seed)}
        req = urllib.request.Request(endpoint.rstrip('/') + '/generate',
            data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=config['timeout_seconds']) as response:
                result = json.load(response)
            finish = result['meta_info']['finish_reason']['type']
            reasoning, answer, verdict = verdict_for(case, result['text'], finish)
            record = {'case_id': case['id'], 'condition': case['condition'], 'replicate': replicate,
                'seed': seed, 'endpoint': endpoint, 'response': result, 'raw_generation': result['text'],
                'reasoning_text': reasoning, 'raw_output': answer, 'verdict': verdict,
                'response_status': 'incomplete' if finish == 'length' else 'completed',
                'input_tokens': result['meta_info']['prompt_tokens'],
                'output_tokens': result['meta_info']['completion_tokens'],
                'reasoning_tokens': None, 'elapsed_seconds': round(time.monotonic() - started, 3)}
            # Compatibility fields for the existing summary, retaining the native response.
            record['native_response'] = result
            record['response'] = {'model': args.model_id, 'reasoning': {'enable_thinking': True},
                'max_output_tokens': config['sampling']['max_new_tokens'],
                'incomplete_details': {'reason': 'max_output_tokens'} if finish == 'length' else None}
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            temp.replace(path)
            print(json.dumps({'saved': path.name, 'finish': finish, 'tokens': record['output_tokens']}), flush=True)
            return record
        except Exception as error:
            failure = {'case_id': case['id'], 'replicate': replicate, 'error': repr(error),
                       'at': datetime.now(timezone.utc).isoformat()}
            with (out / 'service_errors.jsonl').open('a') as f:
                f.write(json.dumps(failure) + '\n')
            raise

    # These are the first formal slots, not additional pilot samples.
    for i, condition in enumerate(config['smoke_conditions']):
        case = next(c for c in suite['cases'] if c['condition'] == condition)
        sample(case, 1, args.endpoints[i % len(args.endpoints)])
    if args.smoke_only:
        return
    with ThreadPoolExecutor(max_workers=len(args.endpoints) * config['concurrency_per_endpoint']) as pool:
        futures = [pool.submit(sample, c, n, args.endpoints[i % len(args.endpoints)])
                   for i, (c, n) in enumerate(jobs) if not (record_dir / f"{c['id']}__{n}.json").exists()]
        errors = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                errors.append(repr(e))
    records = [json.loads(p.read_text()) for p in sorted(record_dir.glob('*.json'))]
    run = {'source_commit': identity['source_commit'], 'status': 'completed' if len(records) == len(jobs) else 'incomplete',
           'api_config': dict(config['sampling'], model=args.model_id, enable_thinking=True),
           'suite_path': str(out / 'suite.json'), 'cases': records, 'service_errors': errors}
    (out / 'run.json').write_text(json.dumps(run, ensure_ascii=False, indent=2))
    # Only summarize after the whole dispatched batch has finished.
    if errors:
        raise RuntimeError(f'{len(errors)} unresolved transport/adapter errors; explicit resume is available')
    analyze_spec = importlib.util.spec_from_file_location('sol_analysis', SOL / 'analyze.py')
    analyze = importlib.util.module_from_spec(analyze_spec)
    sys.modules['run'] = sol
    analyze_spec.loader.exec_module(analyze)
    # Replay truncations and thinking extraction too, then use the unchanged Sol summary.
    for r in records:
        case = next(c for c in suite['cases'] if c['id'] == r['case_id'])
        assert verdict_for(case, r['raw_generation'], r['native_response']['meta_info']['finish_reason']['type'])[2] == r['verdict']
    summary = analyze.summarize(run, suite)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'status': run['status'], 'samples': len(records), 'summary': str(out / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
