"""Complete fixed-tree oracle; relabelling is teacher-only, never a Q input."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from grounded_llm.core.files import atomic_json
from .environment import TokenSwapping


def rank(permutations):
    values=np.asarray(permutations)
    result=np.zeros(values.shape[:-1],dtype=np.int64)
    n=values.shape[-1]
    for i in range(n-1):
        result+=(values[...,i+1:]<values[...,i,None]).sum(axis=-1)*math.factorial(n-1-i)
    return result


def verify_oracle(path,n,edges):
    count=math.factorial(n); m=len(edges)
    states=np.memmap(path/'states.bin',dtype=np.uint8,mode='r',shape=(count,n))
    children=np.memmap(path/'children.bin',dtype='<i4',mode='r',shape=(count,m))
    distances=np.memmap(path/'distances.bin',dtype=np.uint8,mode='r',shape=(count,))
    try:
        if distances[0]!=0 or np.count_nonzero(distances==0)!=1 or np.any(distances==255):
            raise ValueError('Canonical oracle is incomplete or has a false zero')
        for begin in range(0,count,65536):
            ids=np.arange(begin,min(begin+65536,count))
            target=children[ids]; labels=distances[ids].astype(np.int16)
            child_labels=distances[target].astype(np.int16)
            nonzero=ids!=0
            if np.any(child_labels[nonzero].min(axis=1)+1!=labels[nonzero]) or np.any(np.abs(child_labels-labels[:,None])>1):
                raise ValueError('Full-oracle Bellman/edge consistency failed')
            if np.any(children[target,np.arange(m)[None,:]]!=ids[:,None]):
                raise ValueError('Edge swaps are not exact involutions')
        sample=np.random.default_rng(0).choice(count,min(count,4096),replace=False)
        if np.any(rank(states[sample])!=sample): raise ValueError('Permutation rank convention mismatch')
        for edge,(a,b) in enumerate(edges):
            swapped=states[sample].copy(); swapped[:,[a,b]]=swapped[:,[b,a]]
            if np.any(rank(swapped)!=children[sample,edge]): raise ValueError('Incorrect physical edge swap')
        return dict(bellman_states=count,involution_transitions=count*m,physical_swap_samples=len(sample)*m)
    finally:
        for array in (states,children,distances): array._mmap.close()


class TokenCorpus:
    def __init__(self,path):
        self.path=Path(path)
        self.metadata=json.loads((self.path/'oracle.json').read_text())
        self.scope=self.metadata['scope']
        self.env=TokenSwapping(self.scope['vertex_count'],self.scope['edges'])
        self.n=self.env.n; self.size=math.factorial(self.n)
        self.max_candidates=len(self.env.edges); self.input_dim=self.n*self.n
        self.states=np.memmap(self.path/'states.bin',dtype=np.uint8,mode='r',shape=(self.size,self.n))
        self.children=np.memmap(self.path/'children.bin',dtype='<i4',mode='r',shape=(self.size,self.max_candidates))
        self.distances=np.memmap(self.path/'distances.bin',dtype=np.uint8,mode='r',shape=(self.size,))
        graph=np.full((self.n,self.n),100,dtype=np.int32); np.fill_diagonal(graph,0)
        for a,b in self.env.edges: graph[a,b]=graph[b,a]=1
        for middle in range(self.n): graph=np.minimum(graph,graph[:,middle,None]+graph[None,middle,:])
        self.graph_distances=graph

    def close(self):
        for name in ('states','children','distances'):
            getattr(self,name)._mmap.close()

    def features(self,ids):
        return np.eye(self.n,dtype=np.float32)[self.states[np.asarray(ids,dtype=np.int64)]].reshape(*np.shape(ids),self.input_dim)

    def successors(self,ids):
        return self.children[ids].astype(np.int64)

    def distance(self,source,goal,reverse=False):
        source,goal=np.broadcast_arrays(source,goal)
        s,g=self.states[source],self.states[goal]
        inverse=np.argsort(g,axis=-1)
        relative=np.take_along_axis(inverse,s,axis=-1)
        return np.asarray(self.distances[rank(relative)],dtype=np.float32)

    def baseline(self,source,goal):
        source,goal=np.broadcast_arrays(source,goal)
        inverse=np.argsort(self.states[goal],axis=-1)
        destination=np.take_along_axis(inverse,self.states[source],axis=-1)
        vertices=np.arange(self.n)
        return self.graph_distances[vertices,destination].sum(axis=-1)/2

    def describe(self,state_id):
        return dict(scene=0,source=self.scope,state=self.states[state_id].tolist())

    def action(self,state_id,slot): return list(self.env.edges[slot])


def build(config,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'oracle.json').exists(): return load(out)
    scope=json.loads(Path(config['data']['tree']).read_text(encoding='utf-8'))
    n,edges=scope['vertex_count'],scope['edges']
    graph=out/'graph.txt'
    graph.write_text(f'{n} {len(edges)}\n'+'\n'.join(f'{a} {b}' for a,b in edges)+'\n')
    compiler=shutil.which('g++')
    if compiler is None: raise RuntimeError('Full oracle preparation requires g++; inference does not')
    executable=out/'build_oracle'
    subprocess.run([compiler,'-O3','-std=c++17','-march=native',str(Path(__file__).with_name('oracle.cpp')),'-o',str(executable)],check=True)
    started=time.monotonic()
    result=subprocess.run([str(executable),str(graph),str(out)],check=True,capture_output=True,text=True)
    stats=json.loads(result.stdout)
    if stats['visited']!=math.factorial(n): raise ValueError('Incomplete oracle')
    verification=verify_oracle(out,n,edges)
    atomic_json(out/'oracle.json',dict(scope=scope,statistics=stats,seconds=time.monotonic()-started,
                                     rule='one sequential adjacent swap costs1; canonical identity BFS',seed=config['seed'],verification=verification))
    print(json.dumps(dict(task='token_swapping',oracle=stats,seconds=time.monotonic()-started)),flush=True)
    return load(out)


def load(out): return TokenCorpus(out)
