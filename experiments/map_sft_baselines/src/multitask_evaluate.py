"""Dispatch task slices through the existing evaluators and merge their cases."""
import json
from pathlib import Path
import subprocess
import sys
from grounded_llm.core.files import atomic_json
from .metrics import summarize


def evaluate(args, config):
    records=json.loads((args.data/'evaluation.json').read_text())
    if len(records)!=config['evaluation']['tasks']:
        raise ValueError('Mixed evaluation cohort mismatch')
    selected=records[args.start:args.stop]
    if args.mode=='probe':
        raise ValueError('Mixed-task probes require an explicit task-specific study')
    args.out.mkdir(parents=True,exist_ok=True);cases=[]
    for task in dict.fromkeys(r['task'] for r in selected):
        task_records=[r for r in records if r['task']==task]
        ids={r['trajectory_id'] for r in selected if r['task']==task}
        indices=[i for i,r in enumerate(task_records) if r['trajectory_id'] in ids]
        assert indices==list(range(min(indices),max(indices)+1))
        cfg=dict(config,task=task,evaluation=dict(config['evaluation'],tasks=len(task_records)))
        path=args.out/(task+'_config.json');atomic_json(path,cfg)
        output=args.out/task
        command=[sys.executable,'-u','-m','experiments.map_sft_baselines.src.evaluate',
            '--config',str(path),'--data',str(args.data/task),'--model-path',str(args.model_path),
            '--q-checkpoint',str(args.data/task/'qmap.pt'),'--checkpoint',str(args.checkpoint),
            '--out',str(output),'--baseline',args.baseline,'--mode',args.mode,
            '--start',str(min(indices)),'--stop',str(max(indices)+1)]
        subprocess.run(command,check=True)
        metadata={r['trajectory_id']:r for r in task_records}
        rows=[dict(json.loads(line),task=task) for line in (output/'cases.jsonl').read_text().splitlines()]
        for row in rows:
            for field in ('cohort','variant'):
                if field in metadata[row['trajectory_id']]:
                    row[field]=metadata[row['trajectory_id']][field]
        cases.extend(rows)
    lookup={r['trajectory_id']:r for r in cases}
    cases=[lookup[r['trajectory_id']] for r in selected]
    (args.out/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
    atomic_json(args.out/'summary.json',dict(mode=args.mode,baseline=args.baseline,
        checkpoint=json.loads((args.checkpoint/'evaluation_ready.json').read_text()),
        metrics=summarize(cases),by_task={t:summarize([r for r in cases if r['task']==t])
            for t in {r['task'] for r in cases}}))
