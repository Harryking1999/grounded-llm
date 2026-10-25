"""Training-only cycle seeds, complete branches and exact teacher continuations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.model import StateEncoder
from grounded_llm.core.qmap.sampling import CoverageSampler
from .relation_data import keys,materialize,load,sample_goals
from .relation_evaluate import QCache,rollout


def permitted(corpus,data,partition,rows):
    rows=np.asarray(rows,dtype=np.int64).reshape(-1,2)
    children=corpus.successors(rows[:,0])
    goal_allowed=(np.isin(rows[:,1],data['training_goal_pool']) if 'training_goal_pool' in data
                  else np.ones(len(rows),dtype=bool))
    return (goal_allowed & (partition[rows]==0).all(axis=1)&(partition[children]==0).all(axis=1)
        & (rows[:,0]!=rows[:,1])
        & ~np.isin(keys(rows[:,0],rows[:,1],corpus.size),data['reserved_relations'])
        & ~np.isin(keys(children,rows[:,1,None],corpus.size),data['reserved_relations']).any(axis=1))


def cycle_rows(records):
    history={};cycles=[];lengths=[]
    for edge in records:
        for j,q in enumerate(edge['query_index']):
            path=history.setdefault(int(q),[])
            path.append((int(edge['source'][j]),int(edge['goal'][j])))
            if edge['next_repeated'][j]:
                target=int(edge['children'][j,edge['choice'][j]])
                begin=next(i for i,row in enumerate(path) if row[0]==target)
                cycles.extend(path[begin:]);lengths.append(len(path)-begin)
    return np.unique(np.asarray(cycles,dtype=np.int64).reshape(-1,2),axis=0),lengths


def point_pairs(corpus,rows):
    children=corpus.successors(rows[:,0]);goals=np.broadcast_to(rows[:,1,None],children.shape)
    pairs=np.vstack([rows,np.column_stack([children.ravel(),goals.ravel()])])
    return np.unique(pairs[pairs[:,0]!=pairs[:,1]],axis=0)


def expand(corpus,data,partition,seeds,depth,teacher_steps):
    seeds=np.unique(np.asarray(seeds,dtype=np.int64).reshape(-1,2),axis=0)
    safe=permitted(corpus,data,partition,seeds);accepted=seeds[safe]
    if not len(accepted):raise ValueError('No admissible training cycle seeds')
    pieces=[accepted];front=accepted;blocked=[]
    for _ in range(depth):
        child=corpus.successors(front[:,0]);goals=np.broadcast_to(front[:,1,None],child.shape)
        rows=np.unique(np.column_stack([child.ravel(),goals.ravel()]),axis=0)
        rows=rows[rows[:,0]!=rows[:,1]];valid=permitted(corpus,data,partition,rows)
        blocked.append(rows[~valid]);front=rows[valid]
        if not len(front):break
        pieces.append(front)
    child=corpus.successors(accepted[:,0]);goals=np.broadcast_to(accepted[:,1,None],child.shape)
    starts=np.unique(np.column_stack([child.ravel(),goals.ravel()]),axis=0)
    current=starts.copy();active=current[:,0]!=current[:,1]
    reached=~active;interrupted=np.zeros(len(current),bool);steps=np.zeros(len(current),dtype=np.int32)
    for _ in range(teacher_steps):
        ids=np.flatnonzero(active)
        if not len(ids):break
        valid=permitted(corpus,data,partition,current[ids]);bad=ids[~valid]
        interrupted[bad]=True;active[bad]=False
        ids=ids[valid]
        if not len(ids):continue
        rows=current[ids].copy();pieces.append(rows)
        exact=materialize(corpus,rows)
        slot=exact['optimal'].argmax(axis=1)
        if not exact['optimal'][np.arange(len(rows)),slot].all():raise ValueError('Missing exact descent action')
        current[ids,0]=exact['children'][np.arange(len(rows)),slot];steps[ids]+=1
        done=ids[current[ids,0]==current[ids,1]];reached[done]=True;active[done]=False
    branches=np.unique(np.vstack(pieces),axis=0)
    children=corpus.successors(branches[:,0]);following=np.column_stack([children.ravel(),np.broadcast_to(branches[:,1,None],children.shape).ravel()])
    nonterminal=following[:,0]!=following[:,1]
    represented=set(map(tuple,branches.tolist()))
    covered=sum(tuple(row) in represented for row in following[nonterminal].tolist())
    return dict(tasks_seed=accepted,tasks_branches=branches,pairs_seed=point_pairs(corpus,accepted),
        pairs_branches=point_pairs(corpus,branches),teacher_starts=starts,teacher_steps=steps,
        teacher_reached=reached,teacher_blocked=interrupted),dict(raw_seeds=len(seeds),accepted_seeds=len(accepted),
        rejected_seeds=int((~safe).sum()),branch_contexts=len(branches),
        nonterminal_candidate_edges=int(nonterminal.sum()),candidate_edges_with_next_context=covered,
        teacher_paths=len(starts),teacher_paths_reached_goal=int(reached.sum()),
        teacher_paths_blocked_by_holdout=int(interrupted.sum()),teacher_paths_budget_exhausted=int(active.sum()),
        scope='Whole decisions only. Every retained parent has all legal successors; held-out continuations are counted, never pruned or silently labelled as covered.')


def save_bank(corpus,bank,out):
    for kind in ('seed','branches'):
        bank.update({kind+'_'+key:value for key,value in materialize(corpus,bank['tasks_'+kind]).items()})
        bank['pair_distances_'+kind]=corpus.distance(bank['pairs_'+kind][:,0],bank['pairs_'+kind][:,1])
    np.savez_compressed(Path(out)/'branch_bank.npz',**bank)


def combine(corpus,data,partition,banks,depth,teacher_steps):
    seeds=np.unique(np.vstack([bank['tasks_seed'] for bank in banks]),axis=0)
    if not permitted(corpus,data,partition,seeds).all():raise ValueError('Source bank changes the allowed training scope')
    merged,summary=expand(corpus,data,partition,seeds,depth,teacher_steps)
    for key in ('mining_queries','raw_cycle_seeds','switched_goals'):
        merged[key]=np.unique(np.vstack([bank[key] for bank in banks]),axis=0)
    return merged,summary


def merge(config,reference,paths,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    original=json.loads((reference/'config.json').read_text());corpus,data,_=load(original,reference/'data')
    partition=np.load(reference/'data/endpoint_partition.npy');banks=[]
    for path in paths:
        with np.load(path) as archive:
            banks.append({key:archive[key] for key in ('tasks_seed','mining_queries','raw_cycle_seeds','switched_goals')})
    bank,summary=combine(corpus,data,partition,banks,config['branch_depth'],config['teacher_steps'])
    save_bank(corpus,bank,out)
    summary.update(config=config,source_banks=list(map(str,paths)),unique_source_queries=len(bank['mining_queries']),
        scope='Aggregate old and current-policy cycle seeds, preserving all prior roots; recompute complete branches and exact continuations under the same protected training information.')
    atomic_json(out/'bank_summary.json',summary);print(json.dumps(summary),flush=True)


def prepare(config,reference,checkpoint,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    original=json.loads((reference/'config.json').read_text());corpus,data,_=load(original,reference/'data')
    partition=np.load(reference/'data/endpoint_partition.npy');rng=np.random.default_rng(config['seed'])
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    if saved['scope']!=corpus.scope:raise ValueError('Mining model belongs to another tree')
    torch.set_num_threads(config['cpu_threads'])
    model=StateEncoder(input_dim=corpus.input_dim,**saved['config']['model']).to(device)
    model.load_state_dict(saved['model']);model.eval();allowed=data.get('training_goal_pool')
    if allowed is None:allowed=np.flatnonzero(partition==0)
    proposals={} if 'training_goal_pool' in data else dict(distance_bins=config['distance_bins'],proposal='exact_layers')
    seeds=[];lengths=[];queries=[]
    while sum(map(len,queries))<config['queries']:
        n=min(config['mining_batch'],config['queries']-sum(map(len,queries)))
        source=rng.choice(data['tasks_train'][:,0],n*2)
        rows=sample_goals(corpus,source,rng,data['reserved_relations'],ranking=True,allowed=allowed,**proposals)
        distance=corpus.distance(rows[:,0],rows[:,1]);rows=rows[distance>=config['minimum_distance']][:n]
        if not len(rows):continue
        edges=[];rollout(corpus,QCache(model,corpus,device),rows,config['horizon'],'q',edges)
        mined,loop_lengths=cycle_rows(edges);seeds.append(mined);lengths.extend(loop_lengths);queries.append(rows)
        atomic_json(out/'mining_progress.json',dict(queries=sum(map(len,queries)),cycle_relations=sum(map(len,seeds))))
    seeds=np.unique(np.vstack(seeds),axis=0);safe=permitted(corpus,data,partition,seeds);base=seeds[safe]
    switched=[]
    if config.get('goal_switch_attempts') and len(base):
        original_actions=materialize(corpus,base)['optimal'];pending=np.ones(len(base),bool)
        for _ in range(config['goal_switch_attempts']):
            ids=np.flatnonzero(pending)
            if not len(ids):break
            proposed=sample_goals(corpus,base[ids,0],rng,data['reserved_relations'],ranking=True,allowed=allowed,**proposals)
            optimal=materialize(corpus,proposed)['optimal']
            changed=~(original_actions[ids]&optimal).any(axis=1)&(proposed[:,1]!=base[ids,1])
            switched.append(proposed[changed]);pending[ids[changed]]=False
    extra=np.vstack(switched) if switched else np.empty((0,2),dtype=np.int64)
    bank,summary=expand(corpus,data,partition,np.vstack([seeds,extra]),config['branch_depth'],config['teacher_steps'])
    bank['mining_queries']=np.vstack(queries);bank['raw_cycle_seeds']=seeds;bank['switched_goals']=extra
    save_bank(corpus,bank,out)
    summary.update(config=config,source_checkpoint=str(checkpoint),source_checkpoint_step=saved['step'],
        source_checkpoint_commit=saved['source_commit'],original_cycle_seeds=len(seeds),switched_goal_contexts=len(extra),
        cycle_length_counts={str(n):lengths.count(n) for n in sorted(set(lengths))})
    atomic_json(out/'bank_summary.json',summary)
    print(json.dumps(summary),flush=True)


class BranchReplay:
    def __init__(self,corpus,data,partition,bank,options,rng,fit_bank=None):
        self.corpus,self.rng,self.options=corpus,rng,options
        kind=options['kind']
        if kind not in ('seed','branches'):raise ValueError('Expected seed or branches replay')
        self.tasks=bank['tasks_'+kind];self.pairs=bank['pairs_'+kind]
        if not len(self.tasks) or not permitted(corpus,data,partition,self.tasks).all():raise ValueError('Invalid training branch bank')
        expected=point_pairs(corpus,self.tasks)
        if not np.array_equal(self.pairs,expected):raise ValueError('Branch point relations differ from complete candidates')
        self.rank_sampler=CoverageSampler(np.zeros(len(self.tasks)),rng,len(self.tasks),1.)
        self.point_sampler=CoverageSampler(np.zeros(len(self.pairs)),rng,len(self.pairs),1.)
        self.rank_draws=np.zeros(len(self.tasks),dtype=np.int64);self.point_draws=np.zeros(len(self.pairs),dtype=np.int64)
        self.fit={}
        for name in ('seed','branches'):
            rows=(fit_bank if fit_bank is not None else bank)['tasks_'+name]
            if not permitted(corpus,data,partition,rows).all():raise ValueError('Fit bank contains protected queries')
            self.fit.update({key+'_'+name+'_fit':value for key,value in materialize(corpus,rows).items()})

    def apply(self,pair,p,truth,target,tasks,r,point_origin,rank_origin):
        count=round(len(tasks)*self.options['decision_fraction']);slots=self.rng.choice(len(tasks),count,replace=False)
        ri=self.rank_sampler.draw(count);pi=self.point_sampler.draw(self.options['point_count'])
        tasks=tasks.copy();r=r.copy();tasks[slots]=self.tasks[ri];r[slots]=-1;rank_origin[slots]=6
        extra=self.pairs[pi];labels=self.corpus.distance(extra[:,0],extra[:,1])
        np.add.at(self.rank_draws,ri,1);np.add.at(self.point_draws,pi,1)
        return (np.vstack([pair,extra]),np.r_[p,np.full(len(extra),-1,dtype=np.int64)],np.r_[truth,labels],
            np.r_[target,labels],tasks,r,np.r_[point_origin,np.full(len(extra),6,dtype=np.uint8)],rank_origin)

    def state_dict(self):
        return dict(rng=self.rng.bit_generator.state,rank_sampler=self.rank_sampler.state_dict(),
            point_sampler=self.point_sampler.state_dict(),rank_draws=self.rank_draws,point_draws=self.point_draws)

    def load_state_dict(self,state):
        self.rng.bit_generator.state=state['rng'];self.rank_sampler.load_state_dict(state['rank_sampler'])
        self.point_sampler.load_state_dict(state['point_sampler']);self.rank_draws=state['rank_draws'];self.point_draws=state['point_draws']


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--merge-banks',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cpu');args=parser.parse_args()
    config=json.loads(args.config.read_text())
    if args.merge_banks:
        if args.checkpoint:parser.error('Merge uses frozen banks, without another mining checkpoint')
        merge(config,args.reference,args.merge_banks,args.out)
    else:
        if not args.checkpoint:parser.error('Mining requires a frozen checkpoint')
        prepare(config,args.reference,args.checkpoint,args.out,args.device)


if __name__=='__main__':main()
