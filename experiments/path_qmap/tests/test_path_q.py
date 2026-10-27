import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.blocks_distance_map.src.model import BoardEncoder, StateEncoder
from experiments.cml_map_scaling.src.core import action_catalog
from experiments.cube2_qmap.src.environment import Cube2
from experiments.external_map_interface.src.transitions import GraphEnvironment
from experiments.flamingo_map_reader.src.fusion import GatedMapCrossAttention
from experiments.flamingo_map_reader.src.memory import AddressedMapMemoryEncoder, q_map_batch
from experiments.flamingo_map_reader.src.state_q import path_q_step, puzzle_q_batch
from experiments.hanoi_qmap.src.environment import Hanoi
from experiments.path_qmap.src.data import prepare
from experiments.path_qmap.src.evaluate import route_metrics
from experiments.path_qmap.src.run import training_hard_indices, train
from experiments.path_qmap.src.reader import FrozenPathMap
from experiments.puzzle_qmap.src.model import MapReader, MODEL_TYPE, QMap


def cycle_graph(size=32):
    adjacency = np.zeros((size, size), dtype=bool)
    for state in range(size):
        other = (state + 1) % size
        adjacency[state, other] = adjacency[other, state] = True
    return GraphEnvironment(adjacency, action_catalog(adjacency)[0])


class PathQTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads(Path("experiments/path_qmap/configs/path256.json").read_text())
        cls.small = copy.deepcopy(cls.config)
        cls.small["data"].update(validation_tasks_per_distance=1, test_tasks_per_distance=2)

    def test_both_losses_withhold_both_directions_of_all_initial_candidate_relations(self):
        env = cycle_graph()
        data = prepare(env.adjacency, self.small)
        member, children = data["membership"], data["children"]
        np.testing.assert_array_equal(member, member.T)
        self.assertTrue(np.all(member[env.adjacency] == 0))
        for source, goal in data["train_tasks"]:
            self.assertEqual(member[source, goal], 0)
            self.assertTrue(np.all(member[children[source][children[source] >= 0], goal] == 0))
        for name, code in (("validation", 1), ("test", 2)):
            for source, goal in data[f"{name}_tasks"]:
                endpoints = np.r_[source, children[source][children[source] >= 0]]
                self.assertTrue(np.all(member[endpoints, goal] == code))
                self.assertTrue(np.all(member[goal, endpoints] == code))
        self.assertEqual(len(np.unique(data["train_pairs"][:, :2])), len(env.adjacency))

    def test_routing_keeps_return_moves_and_detects_cycles(self):
        env = cycle_graph()
        data = prepare(env.adjacency, self.small)
        source, goal = np.where(~np.eye(len(env.adjacency), dtype=bool))
        tasks = np.column_stack((source, goal))
        exact = route_metrics(data["distances"], data, tasks, 32)
        self.assertEqual(exact["success_rate"], 1.0)
        self.assertEqual(exact["mean_extra_steps_on_solved"], 0)
        collapsed = route_metrics(np.zeros_like(data["distances"]), data, tasks, 32)
        self.assertGreater(collapsed["cycles"], 0)
        self.assertLess(collapsed["success_rate"], 1)

    def test_error_replay_only_uses_unprotected_complete_stars(self):
        data = prepare(cycle_graph().adjacency, self.small)
        self.assertEqual(len(training_hard_indices(data['distances'],data)),0)
        ids = training_hard_indices(np.zeros_like(data['distances']),data)
        self.assertGreater(len(ids),0)
        self.assertTrue(np.all(ids < len(data['train_tasks'])))
        # A protected successor label must be rejected even with a permitted parent.
        source,goal = data['train_tasks'][ids[0]]
        child = data['children'][source,0]
        data['membership'][child,goal] = 2
        with self.assertRaisesRegex(ValueError,'protected complete decision'):
            training_hard_indices(np.zeros_like(data['distances']),data)

    def test_continuation_keeps_parent_eligible_and_publishes_loadable_sampler_state(self):
        env = cycle_graph(); cfg = copy.deepcopy(self.small)
        cfg['training'].update(steps=1,learning_rate=0,eval_every=1,
            hard_replay_fraction=.5,hard_refresh_every=1)
        data = prepare(env.adjacency,cfg)
        model = StateEncoder(128,256,int(data['distances'].max()),'directed_sum',input_dim=32)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared = root/'prepared'; prepared.mkdir()
            np.savez_compressed(prepared/'data.npz',**data,actions=env.actions)
            torch.save(dict(model_type='state_q_path',model=model.state_dict(),optimizer=optimizer.state_dict(),
                cap=model.cap,adjacency=torch.from_numpy(env.adjacency),config=cfg),root/'parent.pt')
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(cfg,root/'run','test',checkpoint=root/'parent.pt',prepared_run=prepared)
            self.assertEqual(result['best_step'],0)
            self.assertEqual(result['initial'],result['final'])
            saved = torch.load(root/'run'/'last.pt',weights_only=True)
            self.assertIsInstance(saved['hard_indices'],torch.Tensor)
            self.assertIn('replay_rng',saved)

    def test_shared_architecture_and_attention_accept_all_task_inputs(self):
        for feature_dim, cls in ((100, BoardEncoder), (9, QMap), (192, QMap), (256, StateEncoder)):
            if cls is QMap:
                model = cls(feature_dim, 128, 256, 14, "linf")
            else:
                model = cls(128, 256, 14, "directed_sum", input_dim=feature_dim)
            linear = [layer for layer in model.network if isinstance(layer, torch.nn.Linear)]
            self.assertEqual([(layer.in_features, layer.out_features) for layer in linear],
                             [(feature_dim, 256), (256, 256), (256, 128)])
            batch = q_map_batch(model.encode(torch.randn(4, feature_dim)))
            reader = AddressedMapMemoryEncoder(128, 32, 6, 16, 4)
            attention = GatedMapCrossAttention(32, 2, 8, key_dim=8, value_dim=16)
            attention.gate.data.fill_(.4)
            hidden = torch.randn(1, 3, 32)
            output = attention(hidden, reader(batch), batch.valid)
            zero = q_map_batch(torch.zeros_like(batch.vectors[0]))
            blank = attention(hidden, reader(zero), zero.valid)
            self.assertEqual(output.shape, hidden.shape)
            self.assertFalse(torch.allclose(output, blank))
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(model.network[0].weight.grad).all())

    def test_frozen_path_checkpoint_supplies_actual_successors_and_checks_topology(self):
        env = cycle_graph(8)
        model = StateEncoder(128, 256, 4, "directed_sum", input_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.pt"
            torch.save({"model_type": "state_q_path", "model": model.state_dict(), "cap": 4,
                        "config": self.config, "adjacency": torch.from_numpy(env.adjacency)}, path)
            reader = FrozenPathMap.load(path)
            self.assertFalse(any(p.requires_grad for p in reader.encoder.parameters()))
            step = path_q_step(env, reader, 1, 6, rng=np.random.default_rng(3))
            self.assertEqual(set(step.candidate_destinations), {0, 2})
            step.map_batch.validate(6)
            for local, destination in enumerate(step.candidate_destinations, 1):
                torch.testing.assert_close(step.map_batch.vectors[0, local + 1], reader.encode([destination])[0])
                self.assertEqual(step.execute(env, local)[1], destination)
            wrong = env.adjacency.copy()
            wrong[0, 2] = wrong[2, 0] = True
            with self.assertRaisesRegex(ValueError, "different graph"):
                path_q_step(GraphEnvironment(wrong, action_catalog(wrong)[0]), reader, 1, 6)

    def test_hanoi_and_cube_use_the_same_q_slot_protocol_without_an_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            for env in (Hanoi(3), Cube2(metric="QTM_URF_6")):
                config = {"dimension": 128, "hidden": 256, "metric": "linf"}
                model = QMap(env.feature_dim, **config, cap=env.diameter)
                path = Path(tmp) / (env.name + ".pt")
                torch.save({"model_type": MODEL_TYPE, "task": env.name, "actions": list(env.actions),
                            "model_config": config, "cap": env.diameter, "model": model.state_dict()}, path)
                reader = MapReader(env, path)
                batch, actions = puzzle_q_batch(reader, 0, 1, rng=np.random.default_rng(2))
                batch.validate(len(env.actions))
                self.assertEqual(batch.vectors.shape, (1, 2 + len(actions), 128))
                for local, action in enumerate(actions, 1):
                    child = env.step(0, action)
                    torch.testing.assert_close(batch.vectors[0, local + 1], reader.encode([child])[0])


if __name__ == "__main__":
    unittest.main()
