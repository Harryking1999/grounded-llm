"""Training-only distance-coordinate targets; readers never load this module."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F
from grounded_llm.core.files import atomic_json
from grounded_llm.tasks.token_swapping.relation_data import rank
from .relation_data import load


def build(corpus,data,partition,out,workers=8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    anchors=np.asarray(data['training_goal_pool'],dtype=np.int64);reserved=data['reserved_relations']
    if not len(anchors) or np.any(np.diff(anchors)<=0):raise ValueError('Sorted distinct training anchors required')
    if (partition[anchors]!=0).any():raise ValueError('Anchors must be training goals')
    if (out/'summary.json').exists():
        if (not np.array_equal(np.load(out/'anchors.npy'),anchors) or not np.array_equal(np.load(out/'reserved.npy'),reserved)
                or not np.array_equal(np.load(out/'partition.npy'),partition)
                or json.loads((out/'summary.json').read_text())['scope']!=corpus.scope):
            raise ValueError('Coordinate-target cache changes protected scope')
        return json.loads((out/'summary.json').read_text())
    temporary=out/'columns.tmp.npy'
    columns=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=(len(anchors),corpus.size))
    def column(j):
        inverse=np.argsort(corpus.states[anchors[j]])
        for begin in range(0,corpus.size,131072):
            end=min(begin+131072,corpus.size)
            columns[j,begin:end]=corpus.distances[rank(inverse[corpus.states[begin:end]])]
        return j
    with ThreadPoolExecutor(max_workers=workers) as executor:list(executor.map(column,range(len(anchors))))
    targets=np.array(columns.T,copy=True,order='C');columns._mmap.close()
    np.save(out/'targets.tmp.npy',targets);(out/'targets.tmp.npy').replace(out/'targets.npy');temporary.unlink()
    allowed=np.ones((corpus.size,len(anchors)),dtype=bool);allowed[partition!=0]=False
    a,b=reserved//corpus.size,reserved%corpus.size
    for source,goal in ((a,b),(b,a)):
        column_ids=np.searchsorted(anchors,goal);valid=column_ids<len(anchors)
        valid&=anchors[np.minimum(column_ids,len(anchors)-1)]==goal
        allowed[source[valid],column_ids[valid]]=False
    np.save(out/'allowed.tmp.npy',allowed);(out/'allowed.tmp.npy').replace(out/'allowed.npy')
    np.save(out/'anchors.npy',anchors);np.save(out/'reserved.npy',reserved);np.save(out/'partition.npy',partition)
    summary=dict(states=corpus.size,coordinates=len(anchors),allowed_labels=int(allowed.sum()),
        protected_labels=int(allowed.size-allowed.sum()),scope=corpus.scope,
        semantics='Exact physical distances to fixed training goals, as supervision only. Protected symmetric relation closures and held input rows masked. Not inference features or an inference lookup map.')
    atomic_json(out/'summary.json',summary);return summary


class AnchorProfiles:
    def __init__(self,path,corpus,data,partition,dimension,device):
        path=Path(path);anchors=np.load(path/'anchors.npy')
        if dimension!=len(anchors) or not np.array_equal(anchors,data['training_goal_pool']):raise ValueError('Map dimensions must match the fixed training anchors')
        if not np.array_equal(np.load(path/'reserved.npy'),data['reserved_relations']):raise ValueError('Protected profile mask differs from current queries')
        if not np.array_equal(np.load(path/'partition.npy'),partition) or json.loads((path/'summary.json').read_text())['scope']!=corpus.scope:
            raise ValueError('Profile input split or physical tree changed')
        self.targets=torch.as_tensor(np.load(path/'targets.npy'),device=device)
        self.allowed=torch.as_tensor(np.load(path/'allowed.npy'),device=device)
        if self.targets.shape!=(corpus.size,dimension) or self.allowed.shape!=self.targets.shape:raise ValueError('Profile cache shape mismatch')

    def loss(self,vectors,ids):
        ids=torch.as_tensor(ids,device=vectors.device)
        targets=self.targets[ids].float();allowed=self.allowed[ids]
        return (F.smooth_l1_loss(vectors,targets,reduction='none')*allowed).sum()/allowed.sum().clamp_min(1)

    @torch.no_grad()
    def probe(self,model,corpus,ids):
        device=next(model.parameters()).device
        vectors=model.encode(torch.as_tensor(corpus.features(ids),device=device))
        index=torch.as_tensor(ids,device=device);allowed=self.allowed[index]
        error=(vectors-self.targets[index].float()).abs()[allowed]
        return dict(states=len(ids),coordinates=int(allowed.sum()),mae=float(error.mean()),
            within_half_step=float((error<=.5).float().mean()),role='Fixed actual training-input coordinate diagnostic; no checkpoint selection.')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();config=json.loads((args.reference/'config.json').read_text())
    corpus,data,_=load(config,args.reference/'data');partition=np.load(args.reference/'data/endpoint_partition.npy')
    print(json.dumps(build(corpus,data,partition,args.out,args.workers)),flush=True)


if __name__=='__main__':main()
