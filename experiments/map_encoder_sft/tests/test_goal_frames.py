"""Every query slot must agree with each adopted reader's pair scoring."""
from pathlib import Path

import numpy as np
import torch

from grounded_llm.tasks.cube2.environment import Cube2
from grounded_llm.tasks.cube2.model import MapReader as CubeReader
from grounded_llm.tasks.puzzle8.environment import Puzzle8
from grounded_llm.tasks.puzzle8.model import MapReader as PuzzleReader


ROOT = Path(__file__).resolve().parents[3]


def test_cube_all_slots_use_the_goal_frame_and_inverse_projection():
    torch.set_num_threads(1)
    env = Cube2(metric='QTM_URF_6')
    reader = CubeReader(env, ROOT / 'tasks/cube2/Qmap/checkpoints/goal_relative_inverse_486.pt',
                        coordinate_frame='goal_relative')
    state, goal = 45, 123
    children = env.successors([state])[0].tolist()
    states = [state, goal, *children]
    q = reader.query_vectors(states, goal)
    torch.testing.assert_close(q, reader.encode(env.relative(goal, states)))
    np.testing.assert_allclose(reader.distance(q, q[1]).numpy(), reader.scores(states, [goal]), atol=1e-5)
    torch.testing.assert_close(q[1], reader.encode([0])[0])


def test_puzzle_all_slots_match_corner_edge_and_center_goal_scores():
    torch.set_num_threads(1)
    env = Puzzle8()
    reader = PuzzleReader(env, ROOT / 'tasks/puzzle8/Qmap/checkpoints/goal_normalized_290000.pt',
                          coordinate_frame='goal_normalized')
    for goal in [0, env.step(0,'Up'), env.step(env.step(0,'Up'),'Left')]:
        state = 45
        children = [int(c) for c in env.successors([state])[0] if c >= 0]
        states = [state, goal, *children]
        q = reader.query_vectors(states, goal)
        framed, targets = env.goal_frame(states, goal)
        torch.testing.assert_close(q, reader.encode(framed))
        np.testing.assert_allclose(reader.distance(q, q[1]).numpy(), reader.scores(states,[goal]), atol=1e-5)
        torch.testing.assert_close(q[1], reader.encode([targets[0]])[0])
