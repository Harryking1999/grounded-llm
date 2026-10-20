import json
from pathlib import Path
import numpy as np
import torch
from grounded_llm.core.reader.numbering import numbered_step
from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.restoration_sft import make_step, demonstration
from experiments.map_sft_baselines.src.restoration_prepare import select_routes


def test_renumber_keeps_physical_action_vector_reason_and_history_aligned():
    env = Cube2(metric="QTM_URF_6")
    start = env.step(env.step(0, "U"), "R")
    middle = env.step(start, "R'")
    # A sparse lookup avoids constructing a full cube Q table in this check.
    class Reader:
        def encode(self, ids):
            return torch.tensor([[float(i), float(i % 7)] for i in ids])
        def distance(self, source, goal):
            return (source - goal).abs().sum(-1)
    reader = Reader()
    steps = [make_step(env, reader, state) for state in [start, middle, 0]]
    record = dict(start=start, goal=0, reference_actions=["R'", "U'"])
    plain = demonstration("cube2", env, record, steps)
    changed = demonstration("cube2", env, record, steps, np.random.default_rng(7))
    assert plain.executed_path == changed.executed_path
    for old, new in zip(plain.turns, changed.turns):
        for i, action in enumerate(new.step.candidate_actions):
            j = old.step.candidate_actions.index(action)
            torch.testing.assert_close(new.step.map_batch.vectors[:, i + 2],
                                       old.step.map_batch.vectors[:, j + 2])
            assert new.step.candidate_destinations[i] == old.step.candidate_destinations[j]
        if old.chosen_id is not None:
            assert old.step.execute(old.chosen_id) == new.step.execute(new.chosen_id)
            assert f"<action>{new.chosen_id}</action>" in new.answer_text
            assert f"{new.chosen_id}: {new.step.execute(new.chosen_id)[0]}" in new.user_text
    assert "Actual executed actions: R'" in changed.turns[1].user_text
    assert changed.turns[-1].answer_text.endswith("<done/>")
    assert "{" not in changed.turns[0].user_text


def test_selection_prefers_length_then_additional_physical_coverage():
    depth = np.array([0, 1, 1, 2, 2, 3])
    successors = np.array([[0], [0], [0], [1], [2], [3]])
    eligible = np.array([False, True, True, True, True, True])
    selected, covered = select_routes(eligible, depth, successors, np.zeros(6, dtype=int), 2, 41)
    assert [states[0] for states, _ in selected] == [5, 4]
    assert covered == {1, 2, 3, 4, 5}


def test_four_conditions_keep_b2_optimizer_and_reader_contract():
    root = Path(__file__).parents[1] / "configs"
    original = json.loads((root / "study.json").read_text())
    for name in ("b2_50k_mlp", "cube2_50k_mlp_renumber", "puzzle8_50k_mlp_renumber"):
        config = json.loads((root / f"{name}.json").read_text())
        assert {k: v for k, v in config["training"].items() if k != "keep_resume_checkpoints"} == original["training"]
        assert {k: v for k, v in config["map"].items() if k != "feature_ffn_hidden_dim"} == original["map"]
        assert config["seed"] == original["seed"]
        assert config["data"]["samples_per_epoch"] == 50000
