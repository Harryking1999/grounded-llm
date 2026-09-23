"""Bounded Step 1 exploration reusing the graph, training and evaluation harness."""
import argparse
import copy
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from .run import ROOT, cases, run_case, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/cml_map_scaling/configs/exploration.json')
    args = parser.parse_args()
    contract = json.loads(args.config.read_text())
    base = json.loads((ROOT / contract['base_config']).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', {'exploration': contract, 'base': base})
    summary = {'complete': False, 'source_revision': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'device': 'CPU', 'cases': []}
    for condition in contract['conditions']:
        config = copy.deepcopy(base)
        config['model']['state_dim'] = condition['state_dim']
        config['training']['method'] = condition['method']
        folder = args.output / condition['name']
        folder.mkdir()
        for spec in cases(config):
            result = run_case(config, spec, folder)
            summary['cases'].append({**condition, **result})
            write_json(args.output / 'summary.json', summary)
    summary['complete'] = True
    write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
