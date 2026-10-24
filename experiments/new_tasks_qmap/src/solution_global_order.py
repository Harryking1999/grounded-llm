"""Read-only dense relation diagnosis; never alters formal planning queries."""
import json
from pathlib import Path
import numpy as np


def prepare_order_bank(corpus, data, per_layer, seed):
    rng=np.random.default_rng(seed);bank={};counts={}
    training_goals=np.unique(data['tasks_train'][:,1])
    for name in ['new_goal_validation','new_layout_goal_validation']:
        goals=np.unique(data['tasks_'+name][:,1])
        if np.intersect1d(training_goals,goals).size:raise ValueError('Diagnostic target is trained')
        rows=[];truth=[]
        for goal in goals:
            sid=corpus.locate(goal);states=np.arange(corpus.offsets[sid],corpus.offsets[sid+1])
            distance=corpus.distance(states,np.full(len(states),goal))
            for lo,hi in [(0,1),(1,2),(2,5),(5,10),(10,20),(20,32),(32,64),(64,np.inf)]:
                pool=np.flatnonzero(np.isfinite(distance)&(distance>lo)&(distance<=hi))
                selected=rng.choice(pool,min(per_layer,len(pool)),replace=False)
                rows.extend((int(states[i]),int(goal)) for i in selected);truth.extend(distance[selected])
        bank['tasks_'+name]=np.asarray(rows,dtype=np.int64)
        bank['distances_'+name]=np.asarray(truth,dtype=np.float32)
        counts[name]=dict(targets=len(goals),queries=len(rows))
    return bank,dict(seed=seed,per_layer=per_layer,counts=counts,
        scope='Supplemental read-only finite-distance diagnostic. Target identities already held out. Formal planning queries and 32-step protocol unchanged. Never used for training feedback or checkpoint selection.')


def global_order_metrics(corpus, rows, truth, prediction):
    records=[]
    for goal in np.unique(rows[:,1]):
        mask=rows[:,1]==goal;t=truth[mask];p=prediction[mask]
        gap=t[None,:]-t[:,None];ordered=p[:,None]<p[None,:]-1e-6
        counts={}
        for label,lo,hi in [('all',0,np.inf),('gap_1',0,1),('gap_2',1,2),('gap_3_5',2,5),('gap_6_20',5,20),('gap_21_plus',20,np.inf)]:
            eligible=(gap>lo)&(gap<=hi);total=int(eligible.sum());correct=int((ordered&eligible).sum())
            counts[label]=dict(correct=correct,comparable=total,accuracy=correct/total if total else None)
        records.append(dict(goal=int(goal),scene=int(corpus.locate(goal)),queries=int(mask.sum()),
            finite_mae=float(np.abs(p-t).mean()),order=counts))
    aggregate={}
    for label in ['all','gap_1','gap_2','gap_3_5','gap_6_20','gap_21_plus']:
        total=sum(r['order'][label]['comparable'] for r in records);correct=sum(r['order'][label]['correct'] for r in records)
        values=[r['order'][label]['accuracy'] for r in records if r['order'][label]['accuracy'] is not None]
        aggregate[label]=dict(correct=correct,comparable=total,accuracy=correct/total if total else None,
            mean_target_accuracy=float(np.mean(values)) if values else None)
    return dict(aggregate=aggregate,by_target=records,
        limitation='Pairs from the same target/layout are dependent; pair counts are not independent sample counts. Close-layer accuracy is reported separately from far-layer ordering.')


def unit_edge_point_coverage(pairs,truth,edges,size,draws=None):
    """Audit directed physical one-step pairs without conflating goal contexts.

    Duplicate pool rows retain all their exposure. A reverse relation is a
    different query; an existing non-unit label is flagged rather than hidden.
    No new relation is added to a training bank.
    """
    pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2);edges=np.asarray(edges,dtype=np.int64).reshape(-1,2)
    truth=np.asarray(truth)
    if len(truth)!=len(pairs) or (draws is not None and len(draws)!=len(pairs)):
        raise ValueError('Point labels/counters do not match pool rows')
    known=np.zeros(len(edges),bool);unit=known.copy()
    exposure=np.zeros(len(edges),np.uint64) if draws is not None else None
    if len(pairs):
        key=pairs[:,0]*np.int64(size)+pairs[:,1];order=np.argsort(key);key=key[order]
        starts=np.r_[0,np.flatnonzero(np.diff(key))+1];unique=key[starts]
        low=np.minimum.reduceat(truth[order],starts);high=np.maximum.reduceat(truth[order],starts)
        query=edges[:,0]*np.int64(size)+edges[:,1];at=np.searchsorted(unique,query)
        safe=np.flatnonzero(at<len(unique));known[safe]=unique[at[safe]]==query[safe]
        ids=np.flatnonzero(known);unit[ids]=(low[at[ids]]==1)&(high[at[ids]]==1)
        if exposure is not None:
            counts=np.add.reduceat(np.asarray(draws,dtype=np.uint64)[order],starts)
            exposure[ids]=counts[at[ids]]
    summary=dict(edges=len(edges),present=int(known.sum()),unit_labels=int(unit.sum()),non_unit_matches=int((known&~unit).sum()),
        unit_exposed=int((unit&(exposure>0)).sum()) if exposure is not None else None)
    return summary,dict(present=known,unit=unit,draws=exposure)


def main():
    import argparse
    import torch
    from grounded_llm.core.qmap.model import StateEncoder
    from .solution_structure import load_corpus
    from .relation_evaluate import QCache
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--per-layer',type=int,default=32)
    parser.add_argument('--seed',type=int,default=2026102409)
    args=parser.parse_args();config=json.loads((args.run/'config.json').read_text());corpus=load_corpus(config,args.run/'data')
    torch.set_num_threads(4)
    if args.prepare:
        if args.bank.exists():raise FileExistsError(args.bank)
        data=dict(np.load(args.run/'data/relations.npz'));bank,description=prepare_order_bank(corpus,data,args.per_layer,args.seed)
        np.savez_compressed(args.bank,**bank);args.bank.with_suffix('.json').write_text(json.dumps(description,indent=2)+'\n');print(description)
    else:
        if args.checkpoint is None or args.out is None:parser.error('--checkpoint and --out required')
        if args.out.exists():raise FileExistsError(args.out)
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']);model.load_state_dict(saved['model']);model.eval();cache=QCache(model,corpus,'cpu')
        bank=dict(np.load(args.bank));results={}
        for name in ['new_goal_validation','new_layout_goal_validation']:
            rows=bank['tasks_'+name];results[name]=global_order_metrics(corpus,rows,bank['distances_'+name],cache.scores(rows[:,0],rows[:,1]))
        args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),step=saved['step'],bank=str(args.bank),results=results),indent=2,allow_nan=False)+'\n')
        print({k:v['aggregate']['all'] for k,v in results.items()})


if __name__=='__main__':main()
