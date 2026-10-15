"""Exact QTM local graphs and freshly posed, layer-balanced training batches.

Only the rule environment and offline labels live here; no learned dynamics.
"""
import json
from pathlib import Path

import numpy as np

from experiments.puzzle_qmap.src.sampling import StratifiedEpochSampler, task_indices
from .data import decorate, evaluation_pairs, pose_groups


def local_template(env):
    """34 unique words, seven parent slots and their complete six-way edges."""
    one = env.successors(0)
    words = np.unique(np.r_[0, one, env.successors(one).ravel()]).astype(np.int32)
    parents = np.searchsorted(words, np.r_[0, one])
    children = np.searchsorted(words, env.successors(words[parents]))
    if len(words) != 34 or not np.array_equal(words[children], env.successors(words[parents])):
        raise ValueError("Expected the complete radius-two QTM graph")
    return words, parents, children


class LocalGraphSampler:
    """Traverse every relative root within its layer; re-pose the entire graph."""
    def __init__(self, env, data, config, rng):
        if env.name != "cube2_qtm":
            raise ValueError("Local graph contract requires the six-action cube")
        self.env, self.rng, self.config = env, rng, config
        self.relatives = data["relative_roots"]
        self.sampler = StratifiedEpochSampler(env.depth[self.relatives], rng)
        self.forbidden = np.zeros(env.size, dtype=bool)
        self.forbidden[data["protected_states"]] = True
        self.words, self.parents, self.children = local_template(env)
        self.local_distances = env.distance(self.words[:, None], self.words[None, :])
        self.groups_drawn = 0
        self.replay = {}
        self.replayed_groups = 0
        self.replay_checks = self.replay_corrected = self.replay_evicted = 0
        self.repair = config["sampling"].get("decision_replay")
        if self.repair and (not 0 < self.repair["fraction"] < 1 or self.repair["capacity"] < 1):
            raise ValueError("Expected a bounded replay buffer and a nonzero fresh fraction")
        self.pair_histogram = np.zeros(env.diameter + 1, dtype=np.int64)
        self.parent_histogram = np.zeros(env.diameter + 1, dtype=np.int64)

    def coverage(self):
        result = {"root_min_layer_passes": self.sampler.passes,
                "root_layers": self.sampler.coverage(), "online_pose_groups": self.groups_drawn,
                "supervised_pair_distance_histogram": self.pair_histogram.tolist(),
                "supervised_parent_distance_histogram": self.parent_histogram.tolist()}
        if self.repair:
            result.update(replayed_pose_groups=self.replayed_groups, replay_buffer_groups=len(self.replay),
                          fresh_pose_groups=self.groups_drawn - self.replayed_groups,
                          replay_checks=self.replay_checks, replay_corrected=self.replay_corrected,
                          replay_evicted_unresolved=self.replay_evicted)
        return result

    def draw(self, groups=None, pair_count=None):
        options = self.config["training"]
        groups = options["graph_batch_size"] if groups is None else groups
        pair_count = options["batch_size"] if pair_count is None else pair_count
        replay_count = (min(len(self.replay), groups - 1, round(groups * self.repair["fraction"]))
                        if self.repair else 0)
        roots = self.relatives[self.sampler.draw(groups - replay_count)]
        nodes = self.env.compose(roots[:, None], self.words[None, :])
        # Column zero is the shared goal. Every other state used in either loss
        # belongs to this whole jointly transformed, split-safe local graph.
        packed = pose_groups(self.env, np.column_stack((np.zeros(len(roots), dtype=int), nodes)),
                             self.forbidden, self.rng, self.config["augmentation"]["max_attempts"])
        if replay_count:
            pool = list(self.replay.values())
            selected = self.rng.choice(len(pool), replay_count, replace=False)
            packed = np.concatenate((packed, np.stack([pool[i] for i in selected])))
            self.replayed_groups += replay_count
        roots = self.env.relative(packed[:, 0], packed[:, 1])
        labels = self.env.depth[self.env.relative(packed[:, :1], packed[:, 1:])]
        width = packed.shape[1]
        offsets = np.arange(groups)[:, None] * width
        parent_labels = labels[:, self.parents]
        live = parent_labels > 0  # A terminal parent does not have an optimal action.
        child_labels = labels[:, self.children]
        delta = child_labels.astype(int) - parent_labels[:, :, None].astype(int)
        if not np.all(np.abs(delta[live]) == 1):
            raise ValueError("QTM successor labels must advance or retreat by one")
        parents = np.broadcast_to(offsets + 1 + self.parents, live.shape)[live]
        goals = np.broadcast_to(offsets, live.shape)[live]
        children = (offsets[:, :, None] + 1 + self.children)[live]

        # Long-range goal calibration and short, composition-aware relations
        # share the same Q evaluations. All labels use the full graph oracle.
        fraction = self.config["sampling"]["local_pair_fraction"]
        if self.config["sampling"].get("complete_goal_distances", False):
            if not 0 <= fraction < 1:
                raise ValueError("Complete goal supervision needs a nonzero goal-pair fraction")
            chosen = np.arange(groups * len(self.words))
            near_count = round(len(chosen) * fraction / (1 - fraction))
        else:
            near_count = round(pair_count * fraction)
            chosen = self.rng.integers(groups * len(self.words), size=pair_count - near_count)
        gi, ni = np.divmod(chosen, len(self.words))
        goal_pairs = np.column_stack((gi * width + ni + 1, gi * width))
        gi = self.rng.integers(groups, size=near_count)
        a = self.rng.integers(len(self.words), size=near_count)
        b = (a + self.rng.integers(1, len(self.words), size=near_count)) % len(self.words)
        near_pairs = np.column_stack((gi * width + a + 1, gi * width + b + 1))
        pair_labels = np.r_[labels.ravel()[chosen], self.local_distances[a, b]]
        self.groups_drawn += groups
        self.pair_histogram += np.bincount(pair_labels, minlength=self.env.diameter + 1)
        self.parent_histogram += np.bincount(parent_labels[live], minlength=self.env.diameter + 1)
        return {"states": packed.ravel(), "pairs": np.concatenate((goal_pairs, near_pairs)),
                "pair_distances": pair_labels,
                "parents": parents, "goals": goals, "children": children,
                "child_distances": child_labels[live], "parent_distances": parent_labels[live],
                "delta": delta[live], "relative_roots": roots,
                "is_replay": np.arange(groups) >= groups - replay_count}

    def observe(self, batch, scores):
        """Training labels only: retain wrong graphs, remove rechecked correct ones."""
        if not self.repair:
            return
        truth = batch["child_distances"]
        chosen = np.asarray(scores).argmin(axis=1)
        wrong = truth[np.arange(len(truth)), chosen] != batch["parent_distances"] - 1
        graph_ids = batch["parents"] // (len(self.words) + 1)
        failed = np.bincount(graph_ids, weights=wrong, minlength=len(batch["relative_roots"])) > 0
        self.replay_checks += int(batch["is_replay"].sum())
        self.replay_corrected += int((batch["is_replay"] & ~failed).sum())
        packed = batch["states"].reshape(-1, len(self.words) + 1)
        for row, keep in zip(packed, failed):
            key = (int(row[0]), int(row[1]))
            if keep:
                self.replay[key] = row.copy()
            else:
                self.replay.pop(key, None)
        while len(self.replay) > self.repair["capacity"]:
            self.replay.pop(next(iter(self.replay)))
            self.replay_evicted += 1


def build_local_graph(env, config, out):
    """Keep exact relative coverage compact; produce poses online, never freeze them."""
    options = config["data"]
    if (env.name != "cube2_qtm" or config["rules"]["metric"] != env.metric or
            config["augmentation"]["kind"] != env.symmetry_kind or
            not config["augmentation"]["enabled"]):
        raise ValueError("Local graph, rules and online symmetry must agree")
    reference, reference_commit = {}, None
    if options.get("reference_data") is not None:
        reference_path = Path(options["reference_data"])
        metadata = json.loads((reference_path.parent / "summary.json").read_text())
        reference_commit = metadata["source_commit"]
        if reference_commit != options["reference_source_commit"]:
            raise ValueError("Wrong reference evaluation states")
        reference = dict(np.load(reference_path, allow_pickle=False))
    protected = np.zeros(env.size, dtype=bool)
    # Preserve endpoints of all old evaluation cohorts, including the earliest
    # reference, without reusing any old action or distance label.
    for key, rows in reference.items():
        if key.startswith(("pairs_", "tasks_")) and any(s in key for s in ("validation", "test")):
            protected[rows[:, :2]] = True
    data, rng = {}, np.random.default_rng(config["seed"])
    buckets = [np.flatnonzero(env.depth == d) for d in range(env.diameter + 1)]
    for split, bucket in (("validation", 1), ("test", 2)):
        rows = evaluation_pairs(env, rng, options[f"pairs_{split}"], bucket, buckets, protected)
        protected[rows[:, :2]] = True
        decorate(env, data, split, rows[:options[f"decisions_{split}"], :2], rows)
    if reference:
        old_pairs = reference["pairs_test"]
        relabeled = np.column_stack((old_pairs[:, :2], env.distance(old_pairs[:, 0], old_pairs[:, 1])))
        # Optional historical comparison: reuse endpoints, recompute QTM labels.
        old_indices = task_indices(env, reference, "test", config["evaluation"]["rollout_count"], config)
        ordered = np.r_[old_indices, np.setdiff1d(np.arange(len(reference["tasks_test"])), old_indices)]
        decorate(env, data, "reference_test", reference["tasks_test"][ordered], relabeled)
        data["reference_rollout_source_indices"] = old_indices
    data["protected_states"] = np.flatnonzero(protected).astype(np.int32)
    data["relative_roots"] = np.arange(1, env.size, dtype=np.int32)
    if options["relative_limit"] is not None:
        # Small smoke contracts still include every distance layer.
        count = options["relative_limit"]
        data["relative_roots"] = np.unique(np.r_[
            [rng.choice(layer) for layer in buckets[1:]],
            rng.choice(data["relative_roots"], count, replace=False)]).astype(np.int32)
    probe = LocalGraphSampler(env, data, config, np.random.default_rng(config["seed"] + 51)).draw(
        options["monitor_groups"], options["monitor_pairs"])
    states = probe["states"]
    pairs = np.column_stack((states[probe["pairs"]], probe["pair_distances"]))
    tasks = np.column_stack((states[probe["parents"]], states[probe["goals"]]))
    decorate(env, data, "train", tasks, pairs)
    if protected[states].any():
        raise ValueError("Protected state entered fixed training monitor")
    words, parents, children = local_template(env)
    summary = {"split_unit": "protected_evaluation_parent_goal_endpoints",
               "states_in_environment": env.size, "action_count": len(env.actions), "diameter": env.diameter,
               "protected_states": int(protected.sum()), "relative_root_types": len(data["relative_roots"]),
               "relative_root_coverage": len(data["relative_roots"]) / (env.size - 1),
               "graph_unique_nodes": len(words), "graph_parent_slots": len(parents),
               "graph_two_step_sequences": int(children[1:].size),
               "online_common_left_multiplication": True,
               "training_arrays_are_fixed_monitor_only": True,
               "reference_source_commit": reference_commit,
               "splits": {s: {"pairs": len(data[f"pairs_{s}"]), "decisions": len(data[f"tasks_{s}"]),
                              "distance_histogram": np.bincount(data[f"pairs_{s}"][:, 2],
                                                                  minlength=env.diameter + 1).tolist()}
                          for s in ("train", "validation", "test", "reference_test")
                          if f"tasks_{s}" in data}}
    np.savez_compressed(out / "data.npz", **data)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return data, summary
