"""Cost-aware greedy policies; no oracle pruning or search at inference."""
import numpy as np
import torch
from grounded_llm.core.qmap.metrics import correlation


@torch.no_grad()
def predictions(model, data, device):
    goal_q = model.encode(torch.as_tensor(data['goals'], device=device))
    output = []
    for begin in range(0, len(data['features']), 8192):
        end = begin + 8192
        vectors = model.encode(torch.as_tensor(data['features'][begin:end], device=device))
        output.append(model.distance(vectors, goal_q[data['goal_ids'][begin:end]]).cpu().numpy())
    return np.concatenate(output)


def rollout(data, values, starts, horizon):
    solved = optimal = cycles = dead = 0
    excess = []
    failures = []
    for start in starts:
        current, cost, seen, trace = int(start), 0., set(), []
        reason = 'budget'
        for _ in range(horizon):
            if data['terminal'][current]: break
            if current in seen:
                cycles += 1; reason = 'cycle'; break
            seen.add(current)
            actions = np.flatnonzero(data['children'][current] >= 0)
            if not len(actions): reason = 'no_legal_action'; break
            children = data['children'][current, actions]
            action = int(actions[np.argmin(data['costs'][current, actions] + values[children])])
            trace.append(action)
            cost += float(data['costs'][current, action])
            current = int(data['children'][current, action])
        success = bool(data['terminal'][current])
        solved += success
        optimal += success and np.isclose(cost, data['distances'][start])
        dead += not np.isfinite(data['distances'][current])
        if success: excess.append(cost - float(data['distances'][start]))
        if not success and len(failures) < 12:
            failures.append({'start': int(start), 'end': current, 'reason': reason, 'actions': trace})
    return {'count': len(starts), 'solved': int(solved), 'optimal': int(optimal),
            'cycles': cycles, 'ended_unreachable': dead,
            'success_rate': solved / len(starts) if len(starts) else None,
            'mean_excess_cost_on_success': float(np.mean(excess)) if excess else None,
            'failure_examples': failures}


def evaluate(data, predicted, config, split):
    region = data['split'] == split
    truth = data['distances']
    reachable = region & np.isfinite(truth)
    decisions = np.flatnonzero(reachable & ~data['terminal'])
    children = data['children'][decisions]
    safe, valid = np.maximum(children, 0), children >= 0
    scores = np.where(valid, data['costs'][decisions] + predicted[safe], np.inf)
    actual = np.where(valid, data['costs'][decisions] + truth[safe], np.inf)
    chosen = scores.argmin(axis=1)
    selected_truth = actual[np.arange(len(decisions)), chosen]
    selected_child = safe[np.arange(len(decisions)), chosen]
    novel = ~data['train_feature_seen'][decisions]
    # Strict ordering includes legal dead ends; ties in the truth are not errors.
    numerator = denominator = 0
    for begin in range(0, len(decisions), 2048):
        a, p, v = actual[begin:begin + 2048], scores[begin:begin + 2048], valid[begin:begin + 2048]
        ordered = (a[:, :, None] < a[:, None, :]) & v[:, :, None] & v[:, None, :]
        numerator += int(((p[:, :, None] < p[:, None, :] - 1e-7) & ordered).sum())
        denominator += int(ordered.sum())
    starts = data['starts'][region[data['starts']] & np.isfinite(truth[data['starts']])]
    error = np.abs(predicted[reachable] - truth[reachable])
    result = {'finite_distance_count': int(reachable.sum()), 'finite_distance_mae': float(error.mean()),
              'finite_distance_spearman': correlation(predicted[reachable], truth[reachable]),
              'decision_count': len(decisions),
              'shortest_action_rate': float(np.isclose(selected_truth, truth[decisions]).mean()),
              'dead_end_choice_rate': float((~np.isfinite(truth[selected_child])).mean()),
              'strict_candidate_order_accuracy': numerator / denominator if denominator else None,
              'novel_state_decisions': int(novel.sum()),
              'novel_state_shortest_action_rate': float(np.isclose(selected_truth[novel], truth[decisions[novel]]).mean()) if novel.any() else None,
              'q': rollout(data, predicted, starts, config['evaluation']['horizon']),
              'oracle': rollout(data, truth, starts, config['evaluation']['horizon']),
              'heuristic': rollout(data, data['baseline'], starts, config['evaluation']['horizon']),
              'immediate_cost_only': rollout(data, np.zeros_like(truth), starts, config['evaluation']['horizon'])}
    dead = region & ~np.isfinite(truth)
    threshold = config['training']['unreachable_target'] - .5
    result['unreachable'] = {'count': int(dead.sum()),
                             'recall_at_proxy_threshold': float((predicted[dead] >= threshold).mean()) if dead.any() else None,
                             'finite_false_positive_rate': float((predicted[reachable] >= threshold).mean())}
    return result
