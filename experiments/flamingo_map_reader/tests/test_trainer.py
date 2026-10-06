"""Exercise a real batched Trainer update and restart, using a tiny CPU Qwen."""

from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, TrainingArguments, set_seed

from experiments.flamingo_map_reader.src.fusion import MapReader
from experiments.flamingo_map_reader.src.graph import graph_step
from experiments.flamingo_map_reader.src.memory import MapMemoryEncoder, JointFeatureMapMemoryEncoder
from experiments.flamingo_map_reader.src.train import MapCollator, MapSFTTrainer, EpochCheckpoint
from experiments.flamingo_map_reader.src.transcript import EncodedTrajectory
from test_graph import line_graph


class TrainerTest(unittest.TestCase):
    def test_checkpointed_chunked_loss_preserves_joint_reader_loss_and_gradients(self):
        environment, qmap = line_graph()
        step = graph_step(environment, qmap, 1, 2, executed_path=[1])
        examples = [(EncodedTrajectory([2, 3, 4, 5, 6], [-100, -100, 4, 5, 6], [0]*5, 3), [step])]
        from experiments.flamingo_map_reader.src.train import collate_examples
        timeline, inputs = collate_examples(examples, 0)
        def make(efficient):
            set_seed(7)
            base = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=1, max_position_embeddings=32, pad_token_id=0))
            base.config.use_cache = False
            reader = MapReader(base, JointFeatureMapMemoryEncoder(2, 32, 2, 8, 4, 16),
                2, head_dim=8, checkpoint_layers=efficient, loss_chunk_tokens=2 if efficient else 0)
            for layer in reader.conditioned_layers:
                layer.map_attention.gate.data.fill_(0.3)
            return reader.train()
        normal, efficient = make(False), make(True)
        left = normal(map_batch=timeline, **inputs).loss
        right = efficient(map_batch=timeline, **inputs).loss
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
        left.backward()
        right.backward()  # Map conditioning has already exited here.
        for (name, p), (_, q) in zip(normal.named_parameters(), efficient.named_parameters()):
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertIsNotNone(q.grad, name)
                torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5, msg=name)
        self.assertGreater(efficient.memory_encoder.feature_ffn[0].weight.grad.norm().item(), 0)

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
                data_collator=MapCollator(0), contract={"test": "batch_resume", "config": {"training": {}}},
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
