"""Exercise a real batched Trainer update and restart, using a tiny CPU Qwen."""

from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments, set_seed

from experiments.flamingo_map_reader.src.fusion import MapReader
from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.memory import MapMemoryEncoder
from experiments.flamingo_map_reader.src.train import MapCollator, MapSFTTrainer, EpochCheckpoint
from experiments.flamingo_map_reader.src.transcript import EncodedTrajectory
from test_graph import line_graph


class TrainerTest(unittest.TestCase):
    def test_standard_batch_save_and_resume(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, executed_path=[1])
        examples = [(EncodedTrajectory([2, 3, 4, 5], [-100, -100, 4, 5], [0]*4, 2), [step])
                    for _ in range(4)]

        def model():
            set_seed(7)
            base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
                intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0))
            base.config.use_cache = False
            return MapReader(base, MapMemoryEncoder(2, 32, 2, 8, 4), 2, head_dim=8)

        def trainer(reader, output):
            args = TrainingArguments(output_dir=str(output), use_cpu=True,
                per_device_train_batch_size=2, gradient_accumulation_steps=1,
                num_train_epochs=1, learning_rate=0.001, save_strategy="no",
                remove_unused_columns=False, label_names=["labels"],
                report_to=[], disable_tqdm=True, seed=11, dataloader_pin_memory=False)
            return MapSFTTrainer(model=reader, args=args, train_dataset=examples,
                data_collator=MapCollator(0), contract={"test": "batch_resume"},
                callbacks=[EpochCheckpoint(0.5)])

        with tempfile.TemporaryDirectory() as directory:
            initial = model()
            before = initial.base_model.get_input_embeddings().weight.detach().clone()
            first = trainer(initial, Path(directory)/"first")
            first.train()
            self.assertEqual(first.state.global_step, 2)
            checkpoint = Path(directory)/"first/checkpoint-1"
            for filename in ("adapter.pt", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"):
                self.assertTrue((checkpoint/filename).exists(), filename)
            self.assertTrue(torch.equal(before, initial.base_model.get_input_embeddings().weight))
            self.assertNotEqual(float(initial.conditioned_layers[0].map_attention.gate), 0.0)
            restarted = model()
            second = trainer(restarted, Path(directory)/"second")
            second.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(second.state.global_step, 2)
            for (_, left), (_, right) in zip(initial.named_parameters(), restarted.named_parameters()):
                torch.testing.assert_close(left, right, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
