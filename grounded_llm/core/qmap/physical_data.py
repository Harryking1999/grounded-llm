"""Compact exact physical graphs and relation splits; task callbacks own rules."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .exact_graph import UnitGraphOracle
from grounded_llm.core.files import atomic_json


def group_seed(key,seed,purpose):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{purpose}:{key}'.encode(),digest_size=8).digest(),'little')


def group_split(key,seed,fractions):
    value=group_seed(key,seed,'split')/2**64
    return ('train','validation','test')[int(np.searchsorted(np.cumsum(fractions),value,side='right'))]


def enumerate_component(env, start, bound):
    states, index, rows = [start], {start: 0}, []
    for state in states:
        children = []
        for _, child, cost in env.transitions(state):
            if cost != 1:
                raise ValueError('This corpus uses unit-cost physical transitions')
            if child not in index:
                if len(states) >= bound:
                    raise RuntimeError('unknown_resource: component exceeds exact enumeration bound')
                index[child] = len(states)
                states.append(child)
            children.append(index[child])
        rows.append(children)
    width = max(map(len, rows))
    packed = np.full((len(states), width), -1, dtype=np.int32)
    for i, row in enumerate(rows): packed[i, :len(row)] = row
    return states, index, packed


def save_scene(path, states, children, goal_ids, metadata, symmetric=False, distances=None, reverse=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.save(path/'states.npy', np.asarray(states, dtype=np.uint8))
    np.save(path/'children.npy', children)
    np.save(path/'goal_ids.npy', np.asarray(goal_ids, dtype=np.int32))
    if distances is None:
        oracle = UnitGraphOracle(children)
        distances = np.stack([oracle.to_goal(g) for g in goal_ids])
        reverse = distances if symmetric else np.stack([oracle.from_goal(g) for g in goal_ids])
    np.save(path/'distances.npy', distances)
    np.save(path/'reverse_distances.npy', reverse)
    metadata = dict(metadata, states=len(states), max_candidates=children.shape[1],
                    finite_max=float(distances[np.isfinite(distances)].max()),
                    finite_counts=np.isfinite(distances).sum(axis=1).tolist(),
                    proven_unreachable=int((~np.isfinite(distances)).sum()), unknown_pairs=0)
    atomic_json(path/'scene.json', metadata)
    return metadata


class PhysicalScene:
    def __init__(self, path, env, decode, baseline):
        self.path, self.env = Path(path), env
        self.metadata = json.loads((self.path/'scene.json').read_text(encoding='utf-8'))
        self.states = np.load(self.path/'states.npy', mmap_mode='r')
        self.children = np.load(self.path/'children.npy', mmap_mode='r')
        self.goals = np.load(self.path/'goal_ids.npy')
        self.distances = np.load(self.path/'distances.npy', mmap_mode='r')
        self.reverse_distances = np.load(self.path/'reverse_distances.npy', mmap_mode='r')
        self.goal_columns = {int(g): i for i, g in enumerate(self.goals)}
        self.goal_column_lookup=np.full(len(self.states),-1,dtype=np.int32)
        self.goal_column_lookup[self.goals]=np.arange(len(self.goals))
        self.decode, self.heuristic = decode, baseline

    def features(self, ids):
        return self.env.features(self.decode(self.states[ids]))

    def distance(self, source, goal):
        source, goal = np.broadcast_arrays(source, goal)
        columns=self.goal_column_lookup[goal]
        if np.any(columns<0): raise ValueError('Oracle goal was not exactly enumerated')
        return np.asarray(self.distances[columns,source],dtype=np.float32)

    def reverse(self, source, goal):
        source, goal = np.broadcast_arrays(source, goal)
        columns=self.goal_column_lookup[goal]
        if np.any(columns<0): raise ValueError('Oracle reverse target was not exactly enumerated')
        return np.asarray(self.reverse_distances[columns,source],dtype=np.float32)


class PhysicalCorpus:
    def __init__(self, scenes):
        self.scenes = scenes
        self.offsets = np.cumsum([0, *[len(scene.states) for scene in scenes]], dtype=np.int64)
        self.size = int(self.offsets[-1])
        self.max_candidates = max(s.children.shape[1] for s in scenes)
        self.input_dim = scenes[0].features(np.array([0])).shape[-1]

    def locate(self, ids):
        return np.searchsorted(self.offsets[1:], ids, side='right')

    def features(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        result = np.empty((*ids.shape, self.input_dim), dtype=np.float32)
        scene_ids = self.locate(ids)
        for scene_id in np.unique(scene_ids):
            mask = scene_ids == scene_id
            result[mask] = self.scenes[scene_id].features(ids[mask]-self.offsets[scene_id])
        return result

    def successors(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        result = np.full((*ids.shape, self.max_candidates), -1, dtype=np.int64)
        scene_ids = self.locate(ids)
        for scene_id in np.unique(scene_ids):
            mask = scene_ids == scene_id
            rows = self.scenes[scene_id].children[ids[mask]-self.offsets[scene_id]]
            result[mask, :rows.shape[-1]] = np.where(rows >= 0, rows+self.offsets[scene_id], -1)
        return result

    def distance(self, source, goal, reverse=False):
        source, goal = np.broadcast_arrays(source, goal)
        scene_ids = self.locate(source)
        if np.any(scene_ids != self.locate(goal)):
            raise ValueError('Distances across different physical scenes are undefined')
        result = np.empty(source.shape, dtype=np.float32)
        for scene_id in np.unique(scene_ids):
            mask = scene_ids == scene_id
            s, g = source[mask]-self.offsets[scene_id], goal[mask]-self.offsets[scene_id]
            method = self.scenes[scene_id].reverse if reverse else self.scenes[scene_id].distance
            result[mask] = method(s, g)
        return result

    def baseline(self, source, goal):
        source, goal = np.broadcast_arrays(source, goal)
        result = np.empty(source.shape, dtype=np.float32)
        scene_ids = self.locate(source)
        for scene_id in np.unique(scene_ids):
            mask = scene_ids == scene_id
            scene = self.scenes[scene_id]
            result[mask] = scene.heuristic(scene.states[source[mask]-self.offsets[scene_id]],
                                           scene.states[goal[mask]-self.offsets[scene_id]])
        return result

    def describe(self, state_id):
        scene_id = int(self.locate(state_id))
        scene = self.scenes[scene_id]
        state = scene.decode(scene.states[[state_id-self.offsets[scene_id]]])[0]
        return dict(scene=scene_id, source=scene.metadata['source'], state=state)

    def action(self, state_id, slot):
        scene_id=int(self.locate(state_id)); scene=self.scenes[scene_id]
        if hasattr(scene.env,'actions'): return scene.env.actions[slot]
        state=scene.decode(scene.states[[state_id-self.offsets[scene_id]]])[0]
        return list(scene.env.transitions(state))[slot][0]
