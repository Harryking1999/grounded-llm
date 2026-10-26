"""Build the explicitly held-out path topology and reuse the pure-Q trainer."""
import argparse
import json
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.path.reader import FrozenPathMap
from grounded_llm.tasks.path.environment import shortest_distances,action_catalog
from experiments.path_qmap.src.run import train


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('config','source-map','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.out.exists():raise ValueError('Use a fresh independent-map output')
    args.out.mkdir(parents=True)
    original=FrozenPathMap.load(args.source_map).adjacency.astype(bool)
    graph=original.copy();rng=np.random.default_rng(cfg['topology']['seed']);swaps=0
    for _ in range(cfg['topology']['maximum_attempts']):
        edges=np.argwhere(np.triu(graph,1));first,second=edges[rng.choice(len(edges),2,replace=False)]
        a,b=map(int,first);c,d=map(int,second)
        if len({a,b,c,d})<4 or graph[a,d] or graph[c,b]:continue
        changed=graph.copy()
        for x,y in [(a,b),(c,d)]:changed[x,y]=changed[y,x]=False
        for x,y in [(a,d),(c,b)]:changed[x,y]=changed[y,x]=True
        try:shortest_distances(changed)
        except ValueError:continue
        graph=changed;swaps+=1
        if swaps==cfg['topology']['edge_swaps']:break
    if swaps!=cfg['topology']['edge_swaps'] or np.array_equal(graph,original):raise ValueError('New topology generation failed')
    assert np.array_equal(graph.sum(0),original.sum(0))
    actions,_=action_catalog(graph);np.savez(args.out/'graph.npz',adjacency=graph,actions=actions)
    suite=dict(cases=[dict(node_count=len(graph),neighbors={str(i):np.flatnonzero(graph[i]).tolist() for i in range(len(graph))})])
    atomic_json(args.out/'suite.json',suite)
    cfg['graph_suite']=str((args.out/'suite.json').resolve())
    atomic_json(args.out/'config.json',cfg)
    result=train(cfg,args.out/'qmap',args.source_commit,device='cpu')
    atomic_json(args.out/'ready.json',dict(source_commit=args.source_commit,node_count=len(graph),edge_swaps=swaps,
        degree_and_density_matched=True,checkpoint=str((args.out/'qmap/best.pt').resolve()),
        scope='New topology and independently trained Qmap; no LLM training on this graph. Capability is reported by the frozen reference.'))


if __name__=='__main__':main()
