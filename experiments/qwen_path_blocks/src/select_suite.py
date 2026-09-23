"""Derive a condition subset from a fixed suite without changing its cases."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--conditions', nargs='+', required=True)
    args = parser.parse_args()
    suite = json.loads(Path(args.suite).read_text())
    requested = set(args.conditions)
    cases = [case for case in suite['cases'] if case['condition'] in requested]
    found = {case['condition'] for case in cases}
    if found != requested:
        raise ValueError(f'Missing requested conditions: {sorted(requested - found)}')
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case IDs in source suite')
    derived = dict(suite, cases=cases)
    Path(args.out).write_text(json.dumps(derived, ensure_ascii=False, indent=2))
    print(json.dumps({'out': args.out, 'conditions': sorted(found),
                      'cases': len(cases), 'calls': sum(case['replicates'] for case in cases)}))


if __name__ == '__main__':
    main()
