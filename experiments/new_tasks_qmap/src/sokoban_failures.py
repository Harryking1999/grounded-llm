"""Training-only cycle exits and irreversible dead branches on a fixed wall."""
import numpy as np
from .relation_data import materialize, training_decision_mask
from .relation_evaluate import QCache


def mine_fixed_wall_failures(model,corpus,data,device,maximum_steps=256):
    if len(corpus.scenes)!=1 or corpus.scenes[0].metadata['split']!='train':
        raise ValueError('Fixed-wall failure mining requires one original training scene')
    scene=corpus.scenes[0];n=corpus.size;ids=np.arange(n)
    children=corpus.successors(ids);valid=children>=0;safe=np.maximum(children,0)
    cache=QCache(model,corpus,device);cache.q(ids)
    critical=[];flags=[];seeds=[]
    report=dict(queries=0,solved=0,cycles=0,immediate_dead_decisions=0,cycle_decisions=0)
    for goal in scene.goals[:scene.metadata['training_goals']]:
        goal=int(goal);rows=np.column_stack([ids,np.full(n,goal)])
        allowed=training_decision_mask(corpus,data,rows)
        truth=corpus.distance(ids,np.full(n,goal));height=cache.scores(ids,np.full(n,goal))
        scores=np.where(valid,height[safe],np.inf)
        nxt=children[ids,scores.argmin(-1)].copy();nxt[nxt<0]=ids[nxt<0];nxt[goal]=goal
        endpoint=nxt.copy();hops=(ids!=goal).astype(np.int64)
        for _ in range(n.bit_length()):hops+=hops[endpoint];endpoint=endpoint[endpoint]
        solved=(endpoint==goal)&(hops<=maximum_steps)
        cycle=(nxt[nxt]==ids)&(nxt!=ids)&allowed
        dead=allowed&~np.isfinite(truth[nxt]);selected=cycle|dead
        take=rows[selected]
        reason=np.where(cycle[selected],np.uint8(8),np.uint8(0))|np.where(dead[selected],np.uint8(2),np.uint8(0))
        reason|=np.where(truth[nxt[selected]]+1!=truth[selected],np.uint8(1),np.uint8(0))
        critical.append(take);flags.append(reason)
        if len(take):
            # One complete optimal continuation per represented goal, while
            # every cycle/dead decision remains in the critical pool.
            order=np.argsort(truth[selected],kind='stable')
            seeds.append(take[order[len(order)//2]])
        report['queries']+=int(allowed.sum());report['solved']+=int((allowed&solved).sum())
        report['cycles']+=int((allowed&~solved).sum())
        report['immediate_dead_decisions']+=int(dead.sum());report['cycle_decisions']+=int(cycle.sum())
    critical=np.concatenate(critical);flags=np.concatenate(flags)
    current=np.asarray(seeds,dtype=np.int64).reshape(-1,2);continuations=[]
    for _ in range(maximum_steps):
        if not len(current):break
        current=current[training_decision_mask(corpus,data,current)]
        if not len(current):break
        continuations.append(current.copy());teacher=materialize(corpus,current)
        choice=teacher['optimal'].argmax(-1)
        current=np.column_stack([teacher['children'][np.arange(len(current)),choice],current[:,1]])
    extra=np.unique(np.concatenate(continuations),axis=0) if continuations else np.empty((0,2),dtype=np.int64)
    tasks=np.unique(np.r_[critical,extra],axis=0)
    keys=tasks[:,0]*np.int64(n)+tasks[:,1]
    reason=np.zeros(len(tasks),np.uint8)
    np.bitwise_or.at(reason,np.searchsorted(keys,critical[:,0]*np.int64(n)+critical[:,1]),flags)
    reason[np.searchsorted(keys,extra[:,0]*np.int64(n)+extra[:,1])]|=16
    if not training_decision_mask(corpus,data,tasks).all():raise ValueError('Failure bank crosses a protected relation')
    report.update(critical_decisions=len(critical),continuation_decisions=len(extra),bank_decisions=len(tasks),
                  full_permitted_training_graph=True,all_legal_candidates_preserved=True)
    return report,dict(**materialize(corpus,tasks),reason=reason)
