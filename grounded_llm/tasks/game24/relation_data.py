"""Exact per-question operation DAGs, with layer-balanced complete targets."""
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.exact_graph import UnitGraphOracle
from grounded_llm.core.qmap.physical_data import PhysicalCorpus,PhysicalScene,save_scene
from .environment import Game24


class GameScene(PhysicalScene):
    def __init__(self,path,record):
        values=[tuple(Fraction(v) for v in row) for row in json.loads((path/'values.json').read_text())]
        self.values=values
        lengths=np.asarray([len(v) for v in values])
        super().__init__(path,Game24(record['max_numbers']),
            lambda rows:[values[int(row[0])] for row in rows],
            lambda s,g:np.maximum(lengths[s[:,0]]-lengths[g[:,0]],0).astype(np.float32))


class GameCorpus(PhysicalCorpus):
    def describe(self,state_id):
        result=super().describe(state_id)
        result['state']=[str(v) for v in result['state']]
        return result


def build(config,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=config['data'];rng=np.random.default_rng(config['seed']);env=Game24(spec['max_numbers'])
    records=[];seen=set()
    scenes=list(spec['scenes'].items())
    if spec.get('additional_training_scenes'):scenes.append(('train',spec['additional_training_scenes']))
    exhaustive=spec.get('initial_sampling')=='exhaustive_distinct'
    if spec.get('initial_sampling','random') not in ('random','exhaustive_distinct'):
        raise ValueError('Unknown initial question sampling')
    if exhaustive:
        alphabet=spec['source_values']
        if alphabet!=sorted(set(alphabet)) or any(type(v) is not int or v<=0 for v in alphabet):
            raise ValueError('Declare distinct ascending positive integer source values')
        roots=list(combinations(alphabet,spec['initial_numbers']))
        if sum(count for split,count in scenes if split=='train')!=len(roots) or any(count for split,count in scenes if split!='train'):
            raise ValueError('Exhaustive finite scope must include every initial combination in training')
        root_order=iter([roots[i] for i in rng.permutation(len(roots))])
    target_values=spec.get('root_targets',[24])
    if not target_values or target_values[0]!=24 or len(set(target_values))!=len(target_values):
        raise ValueError('Keep24 first and use distinct explicit singleton targets')
    targets=[env.canonical([v]) for v in target_values]
    for split,count in scenes:
        for _ in range(count):
            if exhaustive:numbers=next(root_order)
            else:
                while True:
                    numbers=tuple(sorted(rng.integers(spec['minimum_number'],spec['maximum_number']+1,spec['initial_numbers']).tolist()))
                    if numbers not in seen:seen.add(numbers);break
            graph=env.graph(numbers)
            values=[tuple(Fraction(v) for v in row) for row in graph['metadata']['exact_rational_states']]
            children=graph['children']
            for target in targets:
                if target not in values:
                    values.append(target);children=np.vstack([children,np.full((1,len(env.actions)),-1,dtype=np.int32)])
            lengths=np.asarray([len(v) for v in values]);oracle=UnitGraphOracle(children)
            # Interleave layers so both training and held-out pools contain
            # ordinary 2/3-number goals; keep 24 as the first training goal.
            pools=[rng.permutation(np.flatnonzero(lengths==n)).tolist() for n in range(1,spec['initial_numbers'])]
            goals=[values.index(target) for target in targets]
            for i in range(spec['goals_per_scene']*2):
                pool=pools[i%len(pools)]
                while pool:
                    g=pool.pop()
                    if g not in goals and np.isfinite(oracle.to_goal(g)).sum()>=spec['minimum_reachable_per_goal']:
                        goals.append(g);break
                if len(goals)==spec['goals_per_scene']:break
            if len(goals)!=spec['goals_per_scene']:raise ValueError(('Insufficient reachable complete DAG targets',numbers))
            metadata=dict(task='game24',split=split,source=dict(numbers=numbers),max_numbers=spec['max_numbers'],
                original_state=0,training_goals=len(goals)-spec['heldout_goals'],symmetric=False,
                goal_layers=lengths[goals].tolist(),rational_identity='sorted reduced Fraction strings across all questions')
            path=f'scene_{len(records):03d}'
            record=save_scene(out/path,np.arange(len(values))[:,None],children,goals,metadata,state_dtype=np.int32)
            atomic_json(out/path/'values.json',[[str(v) for v in row] for row in values])
            records.append(dict(path=path,**record))
            print(json.dumps(dict(task='game24',scene=len(records),split=split,states=len(values))),flush=True)
    atomic_json(out/'corpus.json',dict(task='game24',scenes=records,seed=config['seed']))
    return load(out)


def load(out):
    out=Path(out);manifest=json.loads((out/'corpus.json').read_text())
    return GameCorpus([GameScene(out/r['path'],r) for r in manifest['scenes']])
