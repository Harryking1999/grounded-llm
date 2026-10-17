"""Find harder task conditions and confirm them on independent fresh questions."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from .difficulty import levels, prepare_level
from .protocol import ROOT
from .report import build
from .run import call, summary, write_json


def latest_records(folder):
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (folder/"records").glob("*.json")]
    return {row["case_id"]:row for row in sorted(rows,key=lambda r:(r["case_id"],r["attempt"]))}


def decision(rows, phase, config):
    fraction = sum(bool(row['verdict'].get('solved')) for row in rows)/len(rows)
    if fraction > config['maximum_solved_fraction']:
        return 'advance'
    return 'confirm' if phase == 'screen' else 'difficulty_confirmed'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--out",required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get('retired'):
        raise ValueError('This queue was retired; see the experiment README for the current scope')
    if min(config['cases_per_level'],config['confirmation_cases']) < 20:
        raise ValueError('Current evaluation requires at least twenty cases per condition and confirmation')
    api = json.loads((ROOT/config["api_config"]).read_text())["api"]
    recovery = json.loads((ROOT/config["service_recovery"]).read_text())
    api.update({key:recovery[key] for key in ("max_service_retries","max_total_service_retries","service_retry_delay_seconds")})
    api["concurrency"] = config["concurrency"]
    out = Path(args.out)
    (out/"records").mkdir(parents=True,exist_ok=True)
    source_commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    suite = json.loads((out/"suite.json").read_text(encoding="utf-8")) if (out/"suite.json").exists() else {
        "config":dict(config,api=api),"source_commit":source_commit,"cases":[]}
    state = json.loads((out/"progress.json").read_text(encoding="utf-8")) if (out/"progress.json").exists() else {
        "started_at":datetime.now(timezone.utc).isoformat(),"source_commit":source_commit,
        "tasks":{task:{"status":"active","level":config['initial_levels'][task]} for task in config["tasks"]}}
    if state.get('stop_policy') != config['policy']:
        state.setdefault('policy_history',[]).append({'source_commit':state.get('policy_source_commit',state['source_commit']),
                                                     'policy':state.get('stop_policy'),'tasks':json.loads(json.dumps(state['tasks']))})
    latest = latest_records(out)
    for item in state['tasks'].values():
        if item['status'] == 'model_error_found':
            item['status'] = 'active'
            item.setdefault('historical_first_errors',[]).append(item.pop('first_error'))
        item.setdefault('phase','screen')
        item.setdefault('level_results',[])
        if item['status'] == 'service_blocked':
            record = latest.get(item.get('case_id'),{})
            if record.get('service_error') and record['attempt'] <= api['max_service_retries']:
                item['status'] = 'active'
    suite['current_policy'] = dict(config,source_commit=source_commit)
    state.update(status='running',api=api,stop_policy=config['policy'],policy_source_commit=source_commit)
    key = os.environ["GND_API_KEY"]

    def save():
        state.update(summary=summary(list(latest.values()),suite["cases"]),updated_at=datetime.now(timezone.utc).isoformat())
        write_json(out/"suite.json",suite)
        write_json(out/"progress.json",state)

    def next_case(task):
        item,options = state['tasks'][task],config['tasks'][task]
        while item['status'] == 'active':
            level,phase = item['level'],item['phase']
            if level >= len(levels(task,options)):
                item['status'] = 'range_exhausted'
                return None
            cases = [case for case in suite['cases'] if case['task'] == task and case['level'] == level
                     and case.get('phase','screen') == phase]
            expected = config['confirmation_cases'] if phase == 'confirm' else config['cases_per_level']
            if cases and len(cases) != expected:
                raise ValueError('Frozen case count differs from the current contract; prepare an expanded suite first')
            if not cases:
                generation = dict(config)
                if phase == 'confirm':
                    generation.update(cases_per_level=config['confirmation_cases'],seed=config['seed']+10000000)
                for retry in range(100):
                    generated = prepare_level(task,level,options,generation)
                    prior = {json.dumps([case['initial'],case['goal']]) for case in suite['cases'] if case['task'] == task}
                    if all(json.dumps([case['initial'],case['goal']]) not in prior for case in generated):
                        break
                    generation['seed'] += 1000000
                else:
                    raise ValueError('Cannot construct independent confirmation set')
                for case in generated:
                    case['phase'] = phase
                    if phase == 'confirm':
                        case['id'] = case['id'].replace('_level','_confirm_level')
                suite['cases'].extend(generated)
                cases = generated
                save()
            for case in cases:
                record = latest.get(case['id'])
                if record and not record.get('service_error'):
                    continue
                if record and record['attempt'] > api['max_service_retries']:
                    item.update(status='service_blocked',case_id=case['id'])
                    return None
                return case
            rows = [latest[case['id']] for case in cases]
            action = decision(rows,phase,config)
            result = {'level':level,'level_value':cases[0]['level_value'],'phase':phase,'valid':len(rows),
                      'solved':sum(bool(row['verdict'].get('solved')) for row in rows),'decision':action,
                      'failures':{name:sum(row['verdict'].get('failure') == name for row in rows)
                                  for name in ('illegal_action','unsolved','parse_error','truncated')}}
            item['level_results'].append(result)
            if action == 'difficulty_confirmed':
                item.update(status=action,confirmed=result)
                return None
            if action == 'advance':
                item.update(level=level+1,phase='screen')
            else:
                item['phase'] = 'confirm'
        return None

    save()
    tasks = list(config['tasks'])
    cursor = 0
    fatal = False
    exhausted = False
    with ThreadPoolExecutor(max_workers=api['concurrency']) as executor:
        active = {}
        while active or any(row['status'] == 'active' for row in state['tasks'].values()):
            if not fatal and not exhausted and not (out/'stop_requested').exists():
                for _ in range(len(tasks)):
                    if len(active) >= api['concurrency']:
                        break
                    task = tasks[cursor % len(tasks)]
                    cursor += 1
                    if task in {case['task'] for case in active.values()}:
                        continue
                    case = next_case(task)
                    if case is None:
                        continue
                    attempt = latest.get(case['id'],{}).get('attempt',0)+1
                    retries = sum(max(0,row['attempt']-1) for row in latest.values()) + sum(
                        bool(latest.get(c['id'])) for c in active.values())
                    if attempt > 1 and retries >= api['max_total_service_retries']:
                        exhausted = True
                        break
                    active[executor.submit(call,case,api,key,attempt)] = case
            state['inflight'] = [case['id'] for case in active.values()]
            save()
            if not active:
                break
            completed,_ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                case = active.pop(future)
                record = future.result()
                write_json(out/'records'/f"{case['id']}_attempt{record['attempt']}.json",record)
                latest[case['id']] = record
                fatal = fatal or record.get('http_status') in (400,401,403,404) or record.get('service_error') in ('returned_model_mismatch','unexpected_output_item')
                print(json.dumps({'case':case['id'],'level_value':case['level_value'],'phase':case.get('phase','screen'),
                                  'attempt':record['attempt'],'verdict':record['verdict']}),flush=True)
                save()
    state['inflight'] = []
    state['status'] = ('api_fatal' if fatal else 'service_budget_exhausted' if exhausted else 'stopped' if (out/'stop_requested').exists()
                       else 'service_incomplete' if any(item['status'] == 'service_blocked' for item in state['tasks'].values())
                       else 'completed_breakpoints_or_ranges')
    save()
    result,text = build(out)
    (out/'summary').mkdir(exist_ok=True)
    write_json(out/'summary/summary.json',result)
    (out/'summary/report.md').write_text(text,encoding='utf-8')


if __name__ == '__main__':
    main()
