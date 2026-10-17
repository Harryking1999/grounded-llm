"""Freeze an expanded difficulty suite and reuse matching completed answers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .difficulty import prepare_level
from .protocol import ROOT, judge
from .run import summary, write_json


def identity(case):
    return json.dumps([case['task'],case['initial'],case['goal'],case['prompt']],sort_keys=True)


def reuse_answers(cases,prior,out):
    progress = json.loads((prior/'progress.json').read_text(encoding='utf-8'))
    if progress['status'] == 'running' or progress.get('inflight'):
        raise RuntimeError('Drain the prior worker before importing answers')
    suite = json.loads((prior/'suite.json').read_text(encoding='utf-8'))
    old_cases = {identity(case):case for case in suite['cases']}
    reused = []
    for case in cases:
        old = old_cases.get(identity(case))
        if not old:
            continue
        records = [(path,json.loads(path.read_text(encoding='utf-8')))
                   for path in (prior/'records').glob(old['id']+'_attempt*.json')]
        if not records:
            continue
        latest = max(records,key=lambda pair:pair[1]['attempt'])[1]
        if latest.get('service_error'):
            continue
        if judge(case,latest['raw_output']) != latest['verdict']:
            raise ValueError('The matching prior answer does not reproduce its saved verdict')
        for path,record in records:
            record = dict(record,case_id=case['id'],reused_from={
                'run':str(prior),'record':path.name,'case_id':old['id'],
                'suite_source_commit':suite['source_commit']})
            write_json(out/'records'/f"{case['id']}_attempt{record['attempt']}.json",record)
        reused.append(latest | {'case_id':case['id']})
    return reused


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('retired') or min(config['cases_per_level'],config['confirmation_cases']) < 20:
        raise ValueError('Use an active configuration with at least twenty cases')
    out = Path(args.out)
    if (out/'suite.json').exists():
        raise FileExistsError('Suite already frozen; resume its worker')
    prior = ROOT/config['prior_screen_run']
    prior_progress = json.loads((prior/'progress.json').read_text(encoding='utf-8'))
    if prior_progress['status'] == 'running' or prior_progress.get('inflight'):
        raise RuntimeError('Drain the prior worker before preparing the expanded suite')
    cases = []
    for task,options in config['tasks'].items():
        group = prepare_level(task,config['initial_levels'][task],options,config)
        for case in group:
            case['phase'] = 'screen'
        cases.extend(group)
    api = json.loads((ROOT/config['api_config']).read_text())['api']
    recovery = json.loads((ROOT/config['service_recovery']).read_text())
    api.update({key:recovery[key] for key in ('max_service_retries','max_total_service_retries','service_retry_delay_seconds')})
    api['concurrency'] = config['concurrency']
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (out/'records').mkdir(parents=True,exist_ok=True)
    reused = reuse_answers(cases,prior,out)
    suite = {'config':dict(config,api=api),'source_commit':source,'cases':cases,
             'reused_valid_answers':len(reused),'prior_run':str(prior)}
    now = datetime.now(timezone.utc).isoformat()
    progress = {'status':'prepared','started_at':now,'updated_at':now,'source_commit':source,
                'api':api,'tasks':{task:{'status':'active','level':config['initial_levels'][task],
                    'phase':'screen','level_results':[]} for task in config['tasks']},
                'inflight':[],'summary':summary(reused,cases),'reused_valid_answers':len(reused)}
    write_json(out/'suite.json',suite)
    write_json(out/'progress.json',progress)
    print(json.dumps({'prepared':len(cases),'reused_valid_answers':len(reused),'out':str(out)}))


if __name__ == '__main__':
    main()
