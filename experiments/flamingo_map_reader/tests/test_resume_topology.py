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

from experiments.flamingo_map_reader.src.train import (
    MapSFTTrainer, pinned_supervision, validate_resume_topology,
    extend_epoch_budget, fixed_warmup_arguments,
)


class ResumeTopologyTest(unittest.TestCase):
    def setUp(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        self.config = json.loads((configs/'blocks_ffn_failure_batch4.json').read_text())
        self.topology = json.loads((configs/'blocks_batch4_two_gpu_resume.json').read_text())

    def test_epoch_extension_keeps_supervision_manifest_and_original_warmup(self):
        configs = Path(__file__).resolve().parents[1] / 'configs'
        source = json.loads((configs/'blocks_kv_restart.json').read_text())
        extension = json.loads((configs/'blocks_kv_twenty_epoch_extension.json').read_text())
        state = dict(global_step=125000, epoch=10.0)
        extended = extend_epoch_budget(source, extension, state, 50000, 4)
        original = dict(config=source, batch_size=1, manifest='unchanged', model_source='same', map_source='same')
        migrated = dict(original, config=extended, batch_size=2, resume_topology=self.topology)
        self.assertEqual(pinned_supervision(original), pinned_supervision(migrated))
        self.assertEqual(source['training']['epochs'], 10)
        self.assertEqual(extended['training']['epochs'], 20)
        self.assertEqual(fixed_warmup_arguments(extended['training'], extension['source_step'])['warmup_steps'], 6250)
        for examples, batch, changed_state in ((50001, 4, state), (50000, 8, state),
                                               (50000, 4, dict(global_step=124000, epoch=9.92))):
            with self.subTest(examples=examples, batch=batch), self.assertRaises(ValueError):
                extend_epoch_budget(source, extension, changed_state, examples, batch)

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
