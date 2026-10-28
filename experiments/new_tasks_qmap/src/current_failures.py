"""Current physical-board failures alongside retained broad relation streams."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from grounded_llm.core.qmap.sampling import CoverageSampler
from .relation_data import FailureReplaySampler, BoardRelationSampler, LocalPathSampler


class CurrentFailureReplay:
    def __init__(self, corpus, data, options, seed, out, state=None):
        self.corpus, self.data, self.options, self.out = corpus, data, options, Path(out)
        self.rng = np.random.default_rng(seed)
        self.coverage_rng = np.random.default_rng(seed+1000003)
        self.coverage_fraction = options.get('coverage_fraction',0.)
        if not 0<=self.coverage_fraction<=1:raise ValueError('Coverage fraction must be between zero and one')
        self.round = 0
        self.last_refresh = options.get('initial_step',0)
        self.total_draws = 0
        if options['contexts_per_batch'] < 1 or options['refresh_every'] < 1:
            raise ValueError('Positive current-failure batch and refresh interval required')
        if state is None:
            with np.load((self.out/options['bank']).resolve()) as archive:
                self.set_bank(dict(archive))
        else:
            self.set_bank(state['bank'])
            self.draws = state['draws'].copy()
            self.rng.bit_generator.state = deepcopy(state['rng'])
            self.round, self.last_refresh, self.total_draws = state['round'], state['last_refresh'], state['total_draws']
            if self.coverage_fraction and state.get('coverage'):
                for sampler,saved in zip(self.coverage_samplers,state['coverage']['samplers']):sampler.load_state_dict(saved)
                self.coverage_rng.bit_generator.state=deepcopy(state['coverage']['rng'])
        self.save_bank()
        self.structural=None
        if options.get('structural_groups'):
            from .rush_transfer import StructuralReplaySampler
            spec=options['structural_groups'];prior=state.get('structural') if state else None
            if prior is None:
                with np.load((self.out/spec['bank']).resolve()) as archive:bank=dict(archive)
            else:bank=prior['bank']
            self.structural=StructuralReplaySampler(corpus,data,bank,spec['groups_per_batch'],seed+2000003,prior)
            if 2*spec['groups_per_batch']>=options['contexts_per_batch']:raise ValueError('Structural pairs must leave current failures in the same quota')
        self.retained=None;self.paths=None;self.last_annotations={}
        if options.get('retained_replay'):
            spec=options['retained_replay'];target=self.out/'retained';target.mkdir(exist_ok=True)
            retained_options={k:v for k,v in options.items() if k not in ('retained_replay','path_groups','structural_groups')}
            retained_options.update(spec,bank=str((self.out/spec['bank']).resolve()))
            self.retained=CurrentFailureReplay(corpus,data,retained_options,seed+3000017,target,
                                               state.get('retained') if state else None)
        if options.get('path_groups'):
            spec=options['path_groups'];prior=state.get('path_groups') if state else None
            if prior is None:
                with np.load((self.out/spec['bank']).resolve()) as archive:self.path_bank=dict(archive)
            else:self.path_bank=prior['bank']
            self.path_rng=np.random.default_rng(seed+4000037)
            self.paths=LocalPathSampler(corpus,data,self.path_bank,spec,self.path_rng,spec['finite_point_fraction'])
            if prior:
                self.paths.load_state_dict(prior['sampler']);self.path_rng.bit_generator.state=deepcopy(prior['rng'])
        maximum_paths=self.paths.options['groups_per_batch']*self.paths.options['maximum_steps'] if self.paths else 0
        retained_count=self.retained.options['contexts_per_batch'] if self.retained else 0
        if maximum_paths+retained_count>=options['contexts_per_batch']:
            raise ValueError('Whole paths and retention must leave current failures in the same supplemental quota')

    def save_bank(self):
        target=self.out/f'current_failure_bank_{self.round:04d}.npz'
        temporary=target.with_suffix('.pending')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**self.bank)
        temporary.replace(target)

    def set_bank(self, bank):
        self.bank = bank
        self.sampler = FailureReplaySampler(self.corpus, self.data, bank, self.options)
        self.draws = np.zeros(len(self.sampler.tasks), dtype=np.uint32)
        if self.coverage_fraction:
            self.coverage_ids=[np.flatnonzero((self.sampler.reason&mask)>0) for mask in (15,16)]
            self.coverage_samplers=[CoverageSampler(np.zeros(len(ids),dtype=np.uint8),self.coverage_rng,len(ids),1.)
                                    for ids in self.coverage_ids]

    def draw_contexts(self,nc,nr):
        parts=[]
        for i,(groups,count) in enumerate(zip((self.sampler.critical,self.sampler.continuations),(nc,nr))):
            coverage=round(count*self.coverage_fraction)
            if coverage:parts.append(self.coverage_ids[i][self.coverage_samplers[i].draw(coverage)])
            parts.append(BoardRelationSampler._draw(groups,count-coverage,self.rng,self.options.get('layouts_per_batch')))
        return np.concatenate(parts)

    def apply(self, pairs, point_indices, tasks, rank_indices, point_index, rank_index,
              point_origin, rank_origin, finite_fraction):
        n = self.options['contexts_per_batch']-(2*self.structural.groups_per_batch if self.structural else 0)
        self.last_annotations={}
        if self.paths:
            if np.any(point_origin) or np.any(rank_origin):
                raise ValueError('Supplemental path groups require the original board streams before other replay')
            pairs,point_indices,tasks,rank_indices,po,ro,_=self.paths.apply(
                pairs,point_indices,tasks,rank_indices,point_index,rank_index,self.path_rng)
            point_origin=point_origin.copy();rank_origin=rank_origin.copy()
            point_origin[po>0]=8;rank_origin[ro>0]=8;n-=int((ro>0).sum())
            self.last_annotations.update({key.replace('rank_path_','rank_walk_push_'):value.copy()
                                          for key,value in self.paths.last_annotations.items()})
        if self.retained:
            pairs,point_indices,tasks,rank_indices,point_origin,rank_origin,ids=self.retained.apply(
                pairs,point_indices,tasks,rank_indices,point_index,rank_index,point_origin,rank_origin,finite_fraction)
            point_origin[point_origin==6]=9;rank_origin[rank_origin==6]=9
            self.last_annotations['rank_retained_index']=ids
            n-=self.retained.options['contexts_per_batch']
        # Existing whole paths occupy a suffix. Keep every original path intact.
        retained = np.flatnonzero(rank_origin != 0)
        limit = int(retained[0]) if len(retained) else len(tasks)
        if n > limit: raise ValueError('Current failures exceed the non-path ranking quota')
        nc = round(n*self.options['critical_fraction'])
        ids = self.draw_contexts(nc,n-nc)
        self.rng.shuffle(ids)
        count = n+int((self.sampler.children[ids]>=0).sum())
        prefix = len(pairs)-round(len(pairs)*finite_fraction)
        available = np.arange(prefix, len(pairs))
        available = available[point_origin[available] == 0]
        truth = self.corpus.distance(pairs[available,0], pairs[available,1])
        positions = np.r_[self.rng.permutation(available[truth>5]),
                          self.rng.permutation(available[(truth>2)&(truth<=5)]),
                          self.rng.permutation(available[truth<=2])][:count]
        if len(positions) != count: raise ValueError('Current failures exceed the remaining finite point quota')
        p, pi, t, ti, po, ro, replay_ids = self.sampler.replace(
            pairs, point_indices, tasks[:limit], rank_indices[:limit], point_index, rank_index,
            ids, np.ones(n,dtype=np.uint8), point_positions=positions)
        tasks, rank_indices = tasks.copy(), rank_indices.copy()
        tasks[:limit], rank_indices[:limit] = t, ti
        point_origin, rank_origin = point_origin.copy(), rank_origin.copy()
        point_origin[po>0] = 6
        rank_origin[:limit][ro>0] = 6
        annotations = np.full(len(tasks), -1, dtype=np.int64)
        annotations[:limit] = replay_ids
        np.add.at(self.draws, ids, 1); self.total_draws += n
        if self.structural:
            p,pi,tasks,rank_indices,point_origin,rank_origin=self.structural.apply(
                p,pi,tasks,rank_indices,point_index,rank_index,point_origin,rank_origin,finite_fraction)
        return p, pi, tasks, rank_indices, point_origin, rank_origin, annotations

    def refresh(self, model, device, config, step):
        if step-self.last_refresh < self.options['refresh_every']: return
        if config['task']=='sokoban':
            from .sokoban_failures import mine_fixed_wall_failures
            summary,bank=mine_fixed_wall_failures(model,self.corpus,self.data,device,
                include_failed_nonoptimal=self.options.get('include_failed_nonoptimal',False))
            if len(bank['tasks']):self.set_bank(bank)
            self.round+=1;self.last_refresh=step;self.save_bank()
            atomic_json(self.out/f'current_failures_{self.round:04d}.json',dict(step=step,**summary))
            return
        from .relation_diagnose import training_start_queries, training_failure_bank, policy_boundaries
        from .relation_evaluate import QCache, planning_ids, rollout
        rows, truth = training_start_queries(self.corpus, self.data, 'original_starts')
        ids = (np.arange(len(rows)) if self.options['queries']=='all' else
               planning_ids(self.corpus, rows, truth, self.options['queries'],
                            np.random.default_rng(config['seed']+51003), [1,2,5,10,20,40,80,256]))
        records = []
        metrics = rollout(self.corpus, QCache(model,self.corpus,device), rows[ids], 256, 'q', edge_records=records)
        _, edges = policy_boundaries(records)
        summary, bank = training_failure_bank(self.corpus,self.data,edges,
            continuation_seeds_per_layout=16,maximum_steps=256)
        if len(bank['tasks']): self.set_bank(bank)
        self.round += 1; self.last_refresh = step
        self.save_bank()
        atomic_json(self.out/f'current_failures_{self.round:04d}.json',dict(
            step=step,solved=metrics['solved'],queries=metrics['count'],cycles=metrics['cycles'],mining=summary))

    def state_dict(self):
        state=dict(bank=self.bank,draws=self.draws.copy(),rng=deepcopy(self.rng.bit_generator.state),
                    round=self.round,last_refresh=self.last_refresh,total_draws=self.total_draws)
        if self.coverage_fraction:
            state['coverage']=dict(rng=deepcopy(self.coverage_rng.bit_generator.state),
                                   samplers=[sampler.state_dict() for sampler in self.coverage_samplers])
        if self.structural:state['structural']=self.structural.state_dict()
        if self.retained:state['retained']=self.retained.state_dict()
        if self.paths:state['path_groups']=dict(bank=self.path_bank,sampler=self.paths.state_dict(),
                                               rng=deepcopy(self.path_rng.bit_generator.state))
        return state

    def report(self):
        result=dict(**self.sampler.summary,refresh_round=self.round,last_refresh=self.last_refresh,
                    total_context_draws=self.total_draws,unique_current_bank_contexts=int((self.draws>0).sum()))
        if self.coverage_fraction:
            result['coverage']=dict(fraction=self.coverage_fraction,
                critical_passes=self.coverage_samplers[0].passes,continuation_passes=self.coverage_samplers[1].passes,
                minimum_critical_draws=int(self.draws[self.coverage_ids[0]].min()),
                minimum_continuation_draws=int(self.draws[self.coverage_ids[1]].min()))
        if self.structural:result['structural_groups']=self.structural.report()
        if self.retained:result['retained_replay']=self.retained.report()
        if self.paths:result['path_groups']=dict(**self.paths.summary,
            coverage_passes=float((self.paths.actual_path_steps/self.paths.lengths).sum()/len(self.paths.paths)),
            minimum_path_replays=int((self.paths.actual_path_steps/self.paths.lengths).min()),
            minimum_path_context_draws=int(self.paths.actual_path_steps.min()),
            unique_paths_drawn=int((self.paths.actual_path_steps>0).sum()))
        return result
