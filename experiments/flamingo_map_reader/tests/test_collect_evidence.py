"""Checkpoint identity comes from published metadata, not spacing of saved steps."""

import json
from pathlib import Path
import tempfile
import unittest

from experiments.flamingo_map_reader.src.collect_evidence import checkpoint_epochs


class EvidenceTest(unittest.TestCase):
    def test_irregular_saves_and_extended_final_keep_their_recorded_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            states = {"checkpoint-128": dict(step=128, epoch=128 / 49000),
                      "checkpoint-196000": dict(step=196000, epoch=4.0),
                      "final": dict(step=245000, epoch=5.0)}
            for name, state in states.items():
                directory = root / "blocks/training/models" / name
                directory.mkdir(parents=True)
                (directory / "evaluation_ready.json").write_text(json.dumps(state))
            (root / "blocks/training/models/checkpoint-245000").mkdir()
            self.assertEqual(checkpoint_epochs(root), {"blocks": states})


if __name__ == "__main__":
    unittest.main()
