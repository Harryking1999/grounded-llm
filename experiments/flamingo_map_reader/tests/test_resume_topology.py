"""Preserve the training objective and global sample groups during 4x1 -> 2x2."""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler
from accelerate.data_loader import BatchSamplerShard

from experiments.flamingo_map_reader.src.train import MapSFTTrainer, pinned_supervision, validate_resume_topology


class ResumeTopologyTest(unittest.TestCase):
    def setUp(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        self.config = json.loads((configs/'blocks_ffn_failure_batch4.json').read_text())
        self.topology = json.loads((configs/'blocks_batch4_two_gpu_resume.json').read_text())

    def test_objective_matches_source_but_other_changes_are_not_normalized(self):
        original = dict(config=self.config, batch_size=1, manifest='same', model_source='same', map_source='same')
        migrated = dict(original, batch_size=2, resume_topology=self.topology)
        self.assertEqual(pinned_supervision(original), pinned_supervision(migrated))
        for change in ('learning_rate', 'gradient_clip_norm'):
            changed = copy.deepcopy(migrated)
            changed['config']['training'][change] *= 2
            self.assertNotEqual(pinned_supervision(original), pinned_supervision(changed))
        with self.assertRaises(ValueError):
            validate_resume_topology(self.config, self.topology, 2, 1)
        with self.assertRaises(ValueError):
            validate_resume_topology(self.config, dict(self.topology, loss_reduction='token_mean'), 2, 2)
        changed = copy.deepcopy(self.config)
        changed['training']['batch_size_candidates_global'] = [8]
        with self.assertRaises(ValueError):
            validate_resume_topology(changed, self.topology, 2, 2)

    def test_global_groups_and_resume_offset_are_identical(self):
        def groups(world, batch):
            ranks = []
            for rank in range(world):
                sampler = RandomSampler(range(40), generator=torch.Generator().manual_seed(17))
                ranks.append(list(BatchSamplerShard(BatchSampler(sampler, batch, False),
                    num_processes=world, process_index=rank)))
            return [[sample for rank_batch in batches for sample in rank_batch] for batches in zip(*ranks)]
        old, new = groups(4, 1), groups(2, 2)
        self.assertEqual(old, new)
        self.assertEqual(old[7:], new[7:])

    def test_rng_restores_active_device_without_modifying_source_checkpoint(self):
        saved = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.random.get_rng_state(),
                     cuda=[torch.tensor([rank], dtype=torch.uint8) for rank in range(4)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rng_state_1.pth'
            torch.save(saved, path)
            trainer = SimpleNamespace(contract=dict(resume_topology=self.topology),
                args=SimpleNamespace(process_index=1, local_process_index=1, device='cuda:1'))
            with mock.patch('torch.cuda.is_available', return_value=True), \
                 mock.patch('torch.cuda.random.set_rng_state') as restore:
                MapSFTTrainer._load_rng_state(trainer, directory)
            self.assertTrue(torch.equal(restore.call_args.args[0], saved['cuda'][1]))
            self.assertEqual(restore.call_args.args[1], 'cuda:1')
            self.assertEqual(len(torch.load(path, weights_only=False)['cuda']), 4)


if __name__ == '__main__':
    unittest.main()
