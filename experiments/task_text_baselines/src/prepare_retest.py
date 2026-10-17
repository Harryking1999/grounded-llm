"""Freeze requested model/prompt retests from existing questions without resampling."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from experiments.sol_dag_blocks.src.tasks import BlocksTask
from .difficulty import harder_prompt
from .protocol import ROOT, judge
from .run import write_json


def retest_cases(config, source):
    if config['samples_per_case'] != 1:
        raise ValueError('Retests use exactly one valid answer per question')
    if config['case_mode'] == 'copy_frozen':
        cases = deepcopy(source['cases'])
    elif config['case_mode'] == 'legacy_blocks12_new_prompt':
        options = source['config']['blocks']
        rules = BlocksTask(options)
        cases = []
        for old in source['cases']:
            if old['condition'] != config['source_condition']:
                continue
            actions = deepcopy(old['construction_reference'])
            case = {'id':old['id'], 'task':'blocks', 'initial':rules.from_grid(old['grid']),
                    'goal':0, 'shape_options':deepcopy(options), 'stratum':'legacy_blocks12',
                    'generation_placements':old['construction_objects'],
                    'reference':{'length':None, 'upper_bound':len(actions), 'actions':actions}}
            case['prompt'] = harder_prompt(case)
            cases.append(case)
    else:
        raise ValueError('Unknown retest case mode')
    if len(cases) != config['expected_cases']:
        raise ValueError('Source question count differs from the committed contract')
    for case in cases:
        verdict = judge(case,json.dumps({'actions':case['reference']['actions'], 'final_status':'solved'}))
        if not verdict['contract_pass']:
            raise ValueError(f"Reference replay failed: {case['id']}")
    return cases


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    source = json.loads((ROOT/config['source_suite']).read_text(encoding='utf-8'))
    cases = retest_cases(config,source)
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'suite.json').exists():
        raise FileExistsError('Suite already frozen; do not overwrite existing questions')
    write_json(args.out/'suite.json',{'config':config,'cases':cases,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'question_source':{'path':config['source_suite'], 'source_commit':source.get('source_commit'),
                           'original_config':source['config']}})
    print(json.dumps({'out':str(args.out),'cases':len(cases),'model':config['api']['model']}))


if __name__ == '__main__':
    main()
