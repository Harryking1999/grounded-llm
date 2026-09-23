"""Derive a content-only result from a completed raw Qwen batch.

This never changes the original run directory.  It writes a compact derived
run and a summary using the same deterministic task environment, but permits
unambiguous array-form paths/actions that differ only in JSON wrapping.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from content_eval import evaluate


ROOT = Path(__file__).resolve().parents[3]
SOL = ROOT / 'experiments/sol_dag_blocks/src'
sys.path.insert(0, str(SOL))
spec = importlib.util.spec_from_file_location('sol_runner', SOL / 'run.py')
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)


KEEP = ('case_id', 'condition', 'replicate', 'seed', 'endpoint', 'response_status',
        'response', 'raw_output', 'verdict', 'input_tokens', 'output_tokens',
        'reasoning_tokens', 'elapsed_seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source = Path(args.run_dir).resolve()
    run = json.loads((source / 'run.json').read_text())
    suite = json.loads((source / 'suite.json').read_text())
    expected = {(case['id'], replicate) for case in suite['cases']
                for replicate in range(1, case['replicates'] + 1)}
    if run.get('status') != 'completed' or len(run.get('cases', [])) != len(expected):
        raise ValueError(f'Content-only scoring requires a completed {len(expected)}-slot raw batch')
    actual = [(record['case_id'], record['replicate']) for record in run['cases']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Raw batch does not cover the fixed slots exactly')
    if run.get('service_errors'):
        raise ValueError('Raw batch has unresolved service errors')
    cases = {case['id']: case for case in suite['cases']}
    records = []
    modes = {}
    for raw_record in run['cases']:
        record = {field: copy.deepcopy(raw_record.get(field)) for field in KEEP if field in raw_record}
        if record.get('response_status') == 'completed':
            record['verdict'] = evaluate(cases[record['case_id']], record.get('raw_output', ''), sol.TASKS)
        mode = record['verdict'].get('content_parse_mode', 'budget_truncated')
        modes[mode] = modes.get(mode, 0) + 1
        records.append(record)
    derived = {'source_commit': run['source_commit'], 'status': 'completed',
               'api_config': run['api_config'], 'suite_path': str(Path(args.out).resolve() / 'suite.json'),
               'cases': records, 'service_errors': [],
               'content_scoring': {'name': 'content_only_unambiguous_json_values',
                                   'source_run': str(source),
                                   'scoring_source_commit': subprocess.check_output(
                                       ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                   'parse_modes': modes}}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'suite.json').write_text(json.dumps(suite, ensure_ascii=False, indent=2))
    (out / 'run.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2))
    # The existing summary remains the authoritative aggregation logic; only
    # its response parser is substituted for this derived condition.
    sol.evaluate = lambda case, raw: evaluate(case, raw, sol.TASKS)
    analyze_spec = importlib.util.spec_from_file_location('sol_analysis', SOL / 'analyze.py')
    analyze = importlib.util.module_from_spec(analyze_spec)
    sys.modules['run'] = sol
    analyze_spec.loader.exec_module(analyze)
    summary = analyze.summarize(derived, suite)
    summary['content_scoring'] = derived['content_scoring']
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({'out': str(out), 'parse_modes': modes,
                      'successes': {name: value['successes'] for name, value in summary['conditions'].items()}}))


if __name__ == '__main__':
    main()
