"""Check selection of published half-epoch weights without launching workers."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.flamingo_map_reader.src.blocks_checkpoint_queue import checkpoints


class CheckpointSelectionTest(unittest.TestCase):
    def publish(self, root, name, step, epoch, adapter=True):
        directory = root / name
        directory.mkdir()
        (directory / 'evaluation_ready.json').write_text(json.dumps(dict(step=step, epoch=epoch)))
        if adapter:
            (directory / 'adapter.pt').touch()
        return directory

    def test_half_epoch_selection_and_latest_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.publish(root, 'checkpoint-0', 0, 0)
            self.publish(root, 'checkpoint-1375', 1375, 0.1)
            self.publish(root, 'checkpoint-6875', 6875, 0.5)
            self.publish(root, 'checkpoint-13750', 13750, 1.0)
            self.publish(root, 'checkpoint-20625', 20625, 1.5, adapter=False)
            self.assertEqual([c['step'] for c in checkpoints(root)], [13750, 6875])

    def test_optimizer_boundary_overshoot_is_allowed_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.publish(root, 'checkpoint-430', 430, 0.5)
            self.publish(root, 'checkpoint-431', 431, 0.5 + 1/860)
            self.publish(root, 'checkpoint-432', 432, 0.5 + 2/860)
            self.assertEqual([c['step'] for c in checkpoints(root)], [431, 430])

    def test_final_deduplicates_same_numbered_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = self.publish(root, 'final', 137500, 10.0)
            self.publish(root, 'checkpoint-137500', 137500, 10.0)
            selected = checkpoints(root)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['path'], final)


if __name__ == '__main__':
    unittest.main()
