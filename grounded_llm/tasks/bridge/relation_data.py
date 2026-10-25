"""Complete weighted personnel graphs; full masks and torch bank are goals."""
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.exact_graph import goal_distances
from grounded_llm.core.qmap.physical_data import PhysicalCorpus,PhysicalScene,save_scene
from .environment import Bridge


def decode(rows): return [int(row[0]) for row in rows]


def build(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=config['data'];rng=np.random.default_rng(config['seed']);records=[]
    seen=set()
    for split,count in spec['scenes'].items():
        for _ in range(count):
            while True:
                times=tuple(sorted(rng.choice(np.arange(1,spec['maximum_time']+1),spec['people'],replace=False).tolist()))
                if times not in seen:seen.add(times);break
            env=Bridge(times);graph=env.graph();children,costs=graph['children'],graph['costs']
            available=np.flatnonzero((children>=0).any(axis=1))
            goals=[env.goal]+rng.choice(available[available!=env.goal],spec['goals_per_scene']-1,replace=False).tolist()
            labels=[]
            for goal in goals:
                labels.append(goal_distances(children,costs,np.arange(len(children))==goal))
            labels=np.stack(labels)
            metadata=dict(task='bridge',split=split,source=dict(times=times),times=times,
                original_state=0,training_goals=len(goals)-spec['heldout_goals'],symmetric=True,
                goal_types=['all_right' if g==env.goal else 'complete_intermediate' for g in goals])
            path=f'scene_{len(records):03d}'
            record=save_scene(out/path,np.arange(len(children))[:,None],children,goals,metadata,
                distances=labels,reverse=labels,costs=costs,state_dtype=np.int32)
            records.append(dict(path=path,**record))
    atomic_json(out/'corpus.json',dict(task='bridge',scenes=records,seed=config['seed']))
    return load(out)


def load(out):
    out=Path(out);manifest=json.loads((out/'corpus.json').read_text())
    return PhysicalCorpus([PhysicalScene(out/r['path'],Bridge(r['times']),decode,
        lambda s,g:np.zeros(len(s),dtype=np.float32)) for r in manifest['scenes']])
