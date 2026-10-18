"""Exact distance-stratified clearable-state indexes derived from existing BDDs.

For original reachable function R and BFS removal layer L_d, a source S is a
reachable state with exact clearing distance d iff R(S) and L_d(full XOR S).
This is a symbolic intersection, not another state-graph generation or search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from .symbolic_graph import SymbolicComponent


def build_component(graph_path, out, maximum_nodes=2000000):
    graph = SymbolicComponent(graph_path)
    level, low, high, counts = [graph.width, graph.width], [0, 1], [0, 1], [0, 1]
    memo, intern = {}, {}

    def intersect(a, b):
        if a == 0 or b == 0: return 0
        if a == 1 and b == 1: return 1
        key = (a, b)
        if key in memo: return memo[key]
        at = min(int(graph.level[a]), int(graph.level[b]))
        al, ah = (int(graph.low[a]), int(graph.high[a])) if graph.level[a] == at else (a, a)
        # Complement the input variables of L_d, not its Boolean truth value.
        bl, bh = (int(graph.high[b]), int(graph.low[b])) if graph.level[b] == at else (b, b)
        left, right = intersect(al, bl), intersect(ah, bh)
        node = (at, left, right)
        if left == right:
            result = left
        elif node in intern:
            result = intern[node]
        else:
            if len(level) >= maximum_nodes:
                raise RuntimeError("Exact clearable index exceeds node budget; no incomplete index published")
            result = len(level);intern[node] = result
            level.append(at);low.append(left);high.append(right)
            counts.append((counts[left] << (level[left] - at - 1)) + (counts[right] << (level[right] - at - 1)))
        memo[key] = result
        return result

    roots = [intersect(graph.root, int(layer)) for layer in graph.layers]
    sizes = [counts[r] << level[r] for r in roots]
    # The construction supplies an empty and a clearable full state.
    assert sizes[0] == 1 and sum(sizes) <= graph.size
    out = Path(out);out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "index.npz", level=np.asarray(level, dtype=np.int16),
             low=np.asarray(low,dtype=np.int32),high=np.asarray(high,dtype=np.int32),
             counts=np.asarray(counts,dtype=np.uint64),roots=np.asarray(roots,dtype=np.int32),
             sizes=np.asarray(sizes,dtype=np.uint64),width=np.asarray(graph.width))
    summary={"complete": True,"clearing_distance_state_counts":sizes,"original_states":graph.size,
             "clearable_states":sum(sizes),"diagram_nodes":len(level)}
    (out/'summary.json').write_text(json.dumps(summary)+'\n')
    return summary


def job(args):
    i, graph, out, maximum = args
    return i, build_component(graph, out, maximum)


class ClearDistanceIndex:
    def __init__(self, path):
        with np.load(Path(path)/'index.npz') as data:
            self.sizes=data['sizes'].copy()
            self.views=[]
            for root,size in zip(data['roots'],self.sizes):
                view=object.__new__(SymbolicComponent)
                for name in ('level','low','high','counts'):
                    setattr(view,name,data[name].copy() if not self.views else getattr(self.views[0],name))
                view.width=int(data['width']);view.root=int(root);view.size=int(size)
                self.views.append(view)

    def sample(self, distance, count, rng):
        view=self.views[int(distance)]
        if not view.size: raise ValueError('Requested empty distance stratum')
        return view.unrank(rng.integers(view.size,size=count))


def prepare(config, out):
    path=Path(config['graph_data']);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    specs=json.loads((path/'graph_specs.json').read_text())
    tasks=[(i,path/'graphs'/str(i),out/str(i),config['maximum_diagram_nodes']) for i in range(len(specs))]
    summaries={}
    with ProcessPoolExecutor(max_workers=config['workers']) as executor:
        futures=[executor.submit(job,args) for args in tasks]
        for future in as_completed(futures):
            i,summary=future.result();summaries[i]=summary
            if len(summaries)%1000==0:print(json.dumps({'complete_components':len(summaries)}),flush=True)
    summary={'complete':True,'components':len(summaries),'graph_data':str(path),
             'component_clearable_states':sum(s['clearable_states'] for s in summaries.values()),
             'scope':'Exact reachable AND clearable subsets indexed by true distance; all original states/negative relations/full legal forks remain in the joint data.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(json.loads(args.config.read_text()),args.out)


if __name__=='__main__':main()
