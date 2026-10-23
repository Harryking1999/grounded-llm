"""Whole exact target basins and training-only policy feedback, without new losses."""
import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from .solution_structure import load_corpus,permitted,context_points
from .relation_data import materialize,keys
from .relation_evaluate import QCache,rollout


def prepare_basins(config,out):
    options=config['data'];source=Path(options['basin_source']).resolve()
    atomic_json(out/'scenes.json',json.loads((source/'scenes.json').read_text()))
    corpus=load_corpus(config,out);old=dict(np.load(source/'relations.npz'))
    reserved=[old['reserved_relations']];held_goals=set()
    for key in old:
        if not key.startswith('tasks_') or key in ('tasks_train','tasks_fit_validation'):continue
        rows=old[key];held_goals.update(map(int,rows[:,1]));cp=context_points(rows,corpus.successors(rows[:,0]))
        reserved.extend([keys(cp[:,0],cp[:,1],corpus.size),keys(cp[:,1],cp[:,0],corpus.size)])
    reserved=np.unique(np.concatenate(reserved));rng=np.random.default_rng(options['structure_seed'])
    original_goals=np.unique(old['tasks_train'][:,1]);goals=list(map(int,original_goals));added=[]
    if options['condition'] in ('C','D'):
        for sid,scene in enumerate(corpus.scenes):
            if scene.metadata['split']!='train':continue
            offset=corpus.offsets[sid];count=0
            # Mix ordinary physical states and states already on training paths.
            pool=np.unique(np.r_[rng.choice(len(scene.states),min(256,len(scene.states)),replace=False)+offset,
                                  old['tasks_train'][corpus.locate(old['tasks_train'][:,0])==sid,0]])
            for goal in rng.permutation(pool):
                if int(goal) in held_goals or int(goal) in goals or goal-offset in scene.goals:continue
                states=np.arange(corpus.offsets[sid],corpus.offsets[sid+1]);labels=corpus.distance(states,np.full(len(states),goal))
                finite=np.isfinite(labels)&(labels>0)
                if finite.sum()<options['minimum_goal_basin'] or labels[finite].max()>=config['training']['unreachable_target']:continue
                goals.append(int(goal));added.append(int(goal));count+=1
                if count==options['additional_goals_per_layout']:break
            if count!=options['additional_goals_per_layout']:raise ValueError(('Insufficient admissible target basins',sid,count))
    contexts=[];coverage=[]
    for goal in goals:
        sid=corpus.locate(goal);states=np.arange(corpus.offsets[sid],corpus.offsets[sid+1]);labels=corpus.distance(states,np.full(len(states),goal))
        finite=np.isfinite(labels)&(labels>0)
        if labels[finite].max()>=config['training']['unreachable_target']:raise ValueError('Exact finite distance exceeds unchanged scale')
        rows=np.column_stack([states[finite],np.full(finite.sum(),goal)]);safe=permitted(corpus,reserved,rows)
        if options['condition']=='A':rows=old['tasks_train'][old['tasks_train'][:,1]==goal];safe=permitted(corpus,reserved,rows)
        accepted=rows[safe];contexts.append(accepted)
        children=corpus.successors(accepted[:,0]);valid=children>=0;truth=np.full(children.shape,np.inf)
        truth[valid]=corpus.distance(children[valid],np.full(valid.sum(),goal));solvable=valid&np.isfinite(truth)&(truth>0)
        covered=np.isin(children,accepted[:,0]);coverage.append(dict(goal=goal,scene=int(sid),basin_states=int(finite.sum()),
            accepted_contexts=len(accepted),protected_exclusions=int((~safe).sum()),finite_edges=int(solvable.sum()),
            finite_edges_with_next_context=int((solvable&covered).sum())))
    contexts=np.unique(np.concatenate(contexts),axis=0);exact=materialize(corpus,contexts)
    pairs=np.unique(context_points(contexts,exact['children']),axis=0);feedback=[];goal_weights={};trace_points=[]
    if options['condition']=='D':
        saved=torch.load(options['feedback_checkpoint'],map_location='cpu',weights_only=False)
        model=StateEncoder(input_dim=corpus.input_dim,**config['model']);model.load_state_dict(saved['model']);model.eval()
        cache=QCache(model,corpus,'cpu');queries=[]
        for goal in goals:
            rows=contexts[contexts[:,1]==goal];labels=corpus.distance(rows[:,0],rows[:,1])
            for lo,hi in [(0,5),(5,10),(10,20),(20,32)]:
                ids=np.flatnonzero((labels>lo)&(labels<=hi))
                if len(ids):queries.append(rows[rng.choice(ids)])
        queries=np.asarray(queries);edges=[];rollout(corpus,cache,queries,32,'q',edges)
        for edge in edges:
            for j,i in enumerate(edge['query_index']):
                goal=int(edge['goal'][j]);slot=edge['choice'][j];after=edge['child_distances'][j,slot];before=edge['parent_distance'][j]
                if np.isfinite(before) and before!=after+1:goal_weights[goal]=options['failed_graph_weight']
            rows=np.column_stack([edge['source'],edge['goal']]);safe=permitted(corpus,reserved,rows)
            if safe.any():trace_points.append(context_points(rows[safe],edge['children'][safe]))
            feedback.append({k:v.tolist() for k,v in edge.items()})
        if trace_points:pairs=np.unique(np.concatenate([pairs,*trace_points]),axis=0)
        # Weight whole target graphs across ALL distance layers, never isolate wrong edges.
        atomic_json(out/'feedback_queries.json',dict(queries=queries.tolist(),weighted_goals=goal_weights,
            checkpoint=options['feedback_checkpoint'],scope='Only permitted training layouts/targets. Labels inspected after unchanged greedy action choice.'))
        # Raw candidate traces belong to ignored runtime data, not the scientific report.
        atomic_json(out/'feedback_traces.json',feedback)
    data={k:v for k,v in old.items() if not k.endswith('_train') and k not in ['terminal_ids','auxiliary_ids']}
    data.update({k+'_train':v for k,v in exact.items()});data.update(pairs_train=pairs,
        pair_distances_train=corpus.distance(pairs[:,0],pairs[:,1]),terminal_ids=np.arange(len(contexts)),
        auxiliary_ids=np.array([],dtype=np.int64),reserved_relations=reserved,
        feedback_goal_weights=np.asarray([goal_weights.get(int(g),1.) for g in contexts[:,1]]))
    np.savez_compressed(out/'relations.npz',**data)
    summary=dict(task='sokoban',condition=options['condition'],finite_max=float(exact['distances'].max()),
        contexts=len(contexts),point_pairs=len(pairs),goals=len(goals),additional_goals=added,coverage=coverage,
        weighted_training_graphs=len(goal_weights),coverage_scope='All finite states to each selected target, except protected relation/candidate boundaries; boundary edges are counted, never claimed closed.',
        sampling='Target-distance strata; policy failures upweight entire target graphs including all layers in D only.')
    atomic_json(out/'summary.json',summary)
    return corpus,data,summary
