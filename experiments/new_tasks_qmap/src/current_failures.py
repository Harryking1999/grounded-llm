"""Current Rush Hour cycle exits alongside the retained broad and path streams."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from grounded_llm.core.files import atomic_json
from .relation_data import FailureReplaySampler, BoardRelationSampler


class CurrentFailureReplay:
    def __init__(self, corpus, data, options, seed, out, state=None):
        self.corpus, self.data, self.options, self.out = corpus, data, options, Path(out)
        self.rng = np.random.default_rng(seed)
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
        self.save_bank()

    def save_bank(self):
        target=self.out/f'current_failure_bank_{self.round:04d}.npz'
        temporary=target.with_suffix('.pending')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**self.bank)
        temporary.replace(target)

    def set_bank(self, bank):
        self.bank = bank
        self.sampler = FailureReplaySampler(self.corpus, self.data, bank, self.options)
        self.draws = np.zeros(len(self.sampler.tasks), dtype=np.uint32)

    def apply(self, pairs, point_indices, tasks, rank_indices, point_index, rank_index,
              point_origin, rank_origin, finite_fraction):
        n = self.options['contexts_per_batch']
        # Existing whole paths occupy a suffix. Keep every original path intact.
        retained = np.flatnonzero(rank_origin != 0)
        limit = int(retained[0]) if len(retained) else len(tasks)
        if n > limit: raise ValueError('Current failures exceed the non-path ranking quota')
        nc = round(n*self.options['critical_fraction'])
        ids = np.r_[BoardRelationSampler._draw(self.sampler.critical, nc, self.rng, self.options.get('layouts_per_batch')),
                    BoardRelationSampler._draw(self.sampler.continuations, n-nc, self.rng, self.options.get('layouts_per_batch'))]
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
        return p, pi, tasks, rank_indices, point_origin, rank_origin, annotations

    def refresh(self, model, device, config, step):
        if step-self.last_refresh < self.options['refresh_every']: return
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
        return dict(bank=self.bank,draws=self.draws.copy(),rng=deepcopy(self.rng.bit_generator.state),
                    round=self.round,last_refresh=self.last_refresh,total_draws=self.total_draws)

    def report(self):
        return dict(**self.sampler.summary,refresh_round=self.round,last_refresh=self.last_refresh,
                    total_context_draws=self.total_draws,unique_current_bank_contexts=int((self.draws>0).sum()))
