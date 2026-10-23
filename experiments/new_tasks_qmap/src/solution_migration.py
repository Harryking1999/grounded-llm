"""Read-only, paired error migration on a fixed complete decision bank."""
import numpy as np
import torch


def decision_outcomes(data, predictions):
    children = data['children_train']; truth = data['child_distances_train']
    valid = children >= 0
    choice = np.where(valid, predictions, np.inf).argmin(axis=1)
    action = truth[np.arange(len(truth)), choice] + 1 == data['distances_train']
    # Equal true costs impose no order. Both finite detours and dead branches
    # participate; an illegal padded slot never participates.
    comparable = valid[:, :, None] & valid[:, None, :] & (truth[:, :, None] < truth[:, None, :])
    ordered = predictions[:, :, None] + 1e-6 < predictions[:, None, :]
    complete = (~comparable | ordered).all(axis=(1, 2))
    return dict(shortest_action=action, complete_candidate_order=complete)


def branch_groups(corpus, data):
    tasks = data['tasks_train']; children = data['children_train']
    truth = data['child_distances_train']; valid = children >= 0
    finite = valid & np.isfinite(truth)
    optimal = finite & (truth + 1 == data['distances_train'][:, None])
    push = np.zeros(valid.shape, bool); reversible = np.zeros(valid.shape, bool)
    scene_ids = corpus.locate(tasks[:, 0])
    for sid in np.unique(scene_ids):
        ids = np.flatnonzero(scene_ids == sid); offset = corpus.offsets[sid]
        scene = corpus.scenes[sid]; legal = valid[ids]
        local_children = children[ids][legal] - offset
        parents = np.broadcast_to(tasks[ids, 0, None] - offset, legal.shape)[legal]
        moved = (scene.states[local_children, 1:] != scene.states[parents, 1:]).any(axis=1)
        back = (scene.children[local_children] == parents[:, None]).any(axis=1)
        p = np.zeros(legal.shape, bool); r = p.copy(); p[legal] = moved; r[legal] = back
        push[ids] = p; reversible[ids] = r
    gap = np.zeros(truth.shape + (truth.shape[1],), dtype=np.float32)
    both = finite[:, :, None] & finite[:, None, :]
    np.subtract(truth[:, None, :], truth[:, :, None], out=gap, where=both)
    return dict(single_legal=valid.sum(axis=1) == 1,
                multiple_optimal=optimal.sum(axis=1) > 1,
                recoverable_suboptimal=(finite & ~optimal).any(axis=1),
                dead_branch=(valid & ~finite).any(axis=1),
                push_available=push.any(axis=1), optimal_push=(push & optimal).any(axis=1),
                suboptimal_push=(push & valid & ~optimal).any(axis=1),
                reversible_edge=reversible.any(axis=1),
                successor_gap_1=((gap == 1) & both).any(axis=(1, 2)),
                successor_gap_2=((gap == 2) & both).any(axis=(1, 2)))


def migration_groups(corpus, data):
    tasks = data['tasks_train']; distances = data['distances_train']
    scenes = corpus.locate(tasks[:, 0])
    return dict(layout={str(int(s)):scenes == s for s in np.unique(scenes)},
                target={str(int(g)):tasks[:, 1] == g for g in np.unique(tasks[:, 1])},
                distance_layer={str(int(d)):distances == d for d in np.unique(distances)},
                branch_type=branch_groups(corpus, data))


def paired_counts(before, after, mask=None):
    if mask is not None: before, after = before[mask], after[mask]
    return dict(count=len(before), old_wrong=int((~before).sum()), new_wrong_total=int((~after).sum()),
                repaired=int((~before & after).sum()), introduced=int((before & ~after).sum()),
                still_wrong=int((~before & ~after).sum()), both_correct=int((before & after).sum()))


def summarize_migration(before, after, groups):
    return {metric:dict(overall=paired_counts(before[metric], after[metric]),
                        **{kind:{key:paired_counts(before[metric], after[metric], mask)
                                  for key, mask in strata.items()} for kind, strata in groups.items()})
            for metric in before}


@torch.no_grad()
def fixed_bank_predictions(model, corpus, data, device, packed=None):
    """Encode each physical endpoint once; identical Q distance and candidates."""
    children = data['children_train']; goals = data['tasks_train'][:, 1]
    valid = children >= 0
    ids = np.unique(np.r_[children[valid], goals])
    if packed is None:
        packed = np.concatenate([np.packbits(corpus.features(ids[start:start+32768]).astype(np.uint8),
                                             axis=1, bitorder='little')
                                 for start in range(0, len(ids), 32768)])
    vectors = []
    for start in range(0, len(ids), 32768):
        features = np.unpackbits(packed[start:start+32768], axis=1, count=corpus.input_dim, bitorder='little')
        vectors.append(model.encode(torch.as_tensor(features, dtype=torch.float32, device=device)))
    vectors = torch.cat(vectors)
    predictions = np.full(children.shape, np.inf, np.float32)
    for start in range(0, len(children), 16384):
        stop = min(start + 16384, len(children)); legal = valid[start:stop]
        source_index = np.searchsorted(ids, children[start:stop][legal])
        goal_index = np.searchsorted(ids, np.broadcast_to(goals[start:stop, None], legal.shape)[legal])
        predictions[start:stop][legal] = model.distance(vectors[source_index], vectors[goal_index]).cpu().numpy()
    return predictions, packed
