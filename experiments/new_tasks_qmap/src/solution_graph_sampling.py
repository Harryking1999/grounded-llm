"""Connected finite graph neighborhoods with layout/target/layer coverage."""
import numpy as np


def training_graph_feedback(corpus,data,model,rng,queries_per_layer=2,fragile_margin=.5):
    """Inspect training rollouts after choice; weight complete target graphs."""
    from .relation_evaluate import QCache,rollout
    tasks=data['tasks_train'];queries=[];goals=np.unique(tasks[:,1])
    for goal in goals:
        ids=np.flatnonzero(tasks[:,1]==goal);distance=data['distances_train'][ids]
        for lo,hi in [(0,5),(5,10),(10,20),(20,32)]:
            pool=ids[(distance>lo)&(distance<=hi)]
            if len(pool):queries.extend(tasks[rng.choice(pool,min(queries_per_layer,len(pool)),replace=False)])
    queries=np.asarray(queries);edges=[];cache=QCache(model,corpus,next(model.parameters()).device)
    rollout(corpus,cache,queries,32,'q',edges)
    counts={int(g):dict(finite_decisions=0,wrong=0,fragile_correct=0) for g in goals}
    for edge in edges:
        for j,g in enumerate(edge['goal']):
            parent=edge['parent_distance'][j]
            if not np.isfinite(parent):continue
            row=counts[int(g)];row['finite_decisions']+=1;choice=edge['choice'][j]
            valid=edge['children'][j]>=0;truth=edge['child_distances'][j];scores=edge['child_predictions'][j]
            optimal=valid&(truth+1==parent)
            if not optimal[choice]:row['wrong']+=1
            else:
                margin=np.min(np.where(valid&~optimal,scores,np.inf))-np.min(np.where(optimal,scores,np.inf))
                row['fragile_correct']+=int(0<=margin<fragile_margin)
    weights={g:1+(2*v['wrong']+v['fragile_correct'])/max(1,v['finite_decisions']) for g,v in counts.items()}
    return np.asarray([weights[int(g)] for g in tasks[:,1]]),dict(by_goal=counts,weights=weights,queries=queries.tolist(),
        scope='Only tasks_train targets and allowed training starts; failure frequency and fragile correct choices inspected after action selection. Whole graphs retain every layer and ordinary background weight at least one.'),edges


class ConnectedGraphSampler:
    def __init__(self,corpus,data,options,rng):
        self.corpus,self.data,self.options,self.rng=corpus,data,options,rng
        tasks=data['tasks_train'];pairs=data['pairs_train'];self.goals=np.unique(tasks[:,1])
        self.layouts=np.unique(corpus.locate(self.goals));self.goals_by_layout={int(s):self.goals[corpus.locate(self.goals)==s] for s in self.layouts}
        self.rows={int(g):np.flatnonzero(tasks[:,1]==g) for g in self.goals}
        self.points={int(g):np.flatnonzero(pairs[:,1]==g) for g in self.goals}
        self.strata={};self.weights={}
        weights=data.get('feedback_goal_weights',np.ones(len(tasks)))
        for g,ids in self.rows.items():
            distance=data['distances_train'][ids]
            self.strata[g]=[ids[(distance>lo)&(distance<=hi)] for lo,hi in [(0,5),(5,10),(10,20),(20,32),(32,np.inf)] if ((distance>lo)&(distance<=hi)).any()]
            self.weights[g]=float(weights[ids[0]])
            if not len(self.points[g]):raise ValueError('Target lacks point supervision')
        self.rank_lookup={int(s)*corpus.size+int(g):i for i,(s,g) in enumerate(tasks)}
        self.point_lookup={int(s)*corpus.size+int(g):i for i,(s,g) in enumerate(pairs)}

    def draw(self,point_count,rank_count):
        from .solution_structure import context_points
        groups=min(self.options.get('graphs_per_batch',4),rank_count)
        layouts=self.rng.choice(self.layouts,groups,replace=groups>len(self.layouts))
        selected=[];point_ids=[]
        for group,sid in enumerate(layouts):
            goals=self.goals_by_layout[int(sid)];weights=np.array([self.weights[int(g)] for g in goals]);weights/=weights.sum()
            goal=int(self.rng.choice(goals,p=weights));quota=rank_count//groups+int(group<rank_count%groups)
            rows=[];seen=set();layers=self.strata[goal]
            while len(rows)<quota:
                layer_index=(int(self.rng.integers(len(layers))) if self.options.get('root_layer_sampling')=='uniform'
                             else len(rows)%len(layers))
                layer=layers[layer_index];eligible=layer[~np.isin(layer,list(seen))]
                if not len(eligible):eligible=self.rows[goal][~np.isin(self.rows[goal],list(seen))]
                if not len(eligible):seen.clear();eligible=layer
                root=int(self.rng.choice(eligible));pending=[root];local=0
                while pending and len(rows)<quota and local<self.options.get('neighborhood_size',8):
                    index=pending.pop(0)
                    if index in seen:continue
                    rows.append(index);seen.add(index);local+=1
                    for child in self.data['children_train'][index]:
                        neighbor=self.rank_lookup.get(int(child)*self.corpus.size+goal) if child>=0 else None
                        if neighbor is not None and neighbor not in seen:pending.append(neighbor)
                # Tiny basins may have fewer decisions than a batch quota.
                if len(seen)==len(self.rows[goal]):seen.clear()
            selected.extend(rows)
            budget=point_count//groups+int(group<point_count%groups)
            point_ids.extend(self.rng.choice(self.points[goal],budget).tolist())
        r=np.asarray(selected,dtype=np.int64);p=np.asarray(point_ids,dtype=np.int64)
        self.rng.shuffle(p)
        rows=self.data['tasks_train'][r];required=context_points(rows,self.data['children_train'][r])
        if len(required)>point_count:raise ValueError('Complete candidate supervision exceeds matched point budget')
        p[:len(required)]=[self.point_lookup[int(s)*self.corpus.size+int(g)] for s,g in required]
        return p,r
