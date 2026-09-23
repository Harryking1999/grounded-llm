"""Authority for output parsing, external judging and metrics; no model dependency."""
import re
from collections import Counter
import numpy as np

SCORE_KEYS = {'answer', 'parsed', 'format_valid', 'node_id_valid', 'graph_distance',
              'target', 'correct', 'action_legal', 'decision', 'valid'}

class ScoringMismatch(RuntimeError):
    """Stop immediately when stored and recomputed verdicts disagree."""

def parse_output(output, node_count, scorer='strict_node'):
    """Return a parsed prediction; no graph adjacency or reference labels."""
    if scorer == 'strict_node':
        return dict(raw_output=output, **parse_answer(output, node_count))
    if scorer == 'semantic_action':
        return dict(raw_output=output, decision=extract_decision(output))
    if scorer == 'choice':
        return dict(output=output)
    raise ValueError(f'Unknown parser: {scorer}')

def score_prediction(prediction, example, adjacency, distances, scorer='strict_node'):
    """Score a parsed prediction using only the example and environment truth."""
    if scorer == 'strict_node':
        return score_node(dict(example, **prediction), adjacency, distances)
    if scorer == 'semantic_action':
        return score_decision(example, prediction['raw_output'], adjacency, distances, prediction['decision'])
    if scorer == 'choice':
        return dict(example, **prediction, valid=prediction['output'] in example['options'], correct=prediction['output'] in example['answers'])
    raise ValueError(f'Unknown scorer: {scorer}')

def score_record(row, adjacency, distances, scorer, verify_existing=False, decision=None):
    """Recompute from raw output, without letting old score fields leak in."""
    item = {key: value for key, value in row.items() if key not in SCORE_KEYS}
    if scorer == 'choice':
        scored = score_prediction(parse_output(row['output'], len(adjacency), scorer), item, adjacency, distances, scorer)
    elif scorer == 'semantic_action':
        scored = score_decision(item, row['raw_output'], adjacency, distances, decision)
    else:
        scored = score_prediction(parse_output(row['raw_output'], len(adjacency), scorer), item, adjacency, distances, scorer)
    if verify_existing:
        if 'correct' not in row:
            raise ValueError('Verification requires stored correctness for every prediction')
        for key in sorted(SCORE_KEYS & row.keys()):
            if key not in scored or row[key] != scored[key]:
                identity = {k: row[k] for k in ('condition', 'checkpoint', 'template', 'task', 'u', 'g', 'id') if k in row}
                raise ScoringMismatch(f'{identity}: {key}: old={row[key]!r}, new={scored.get(key)!r}')
    return scored

def metrics(rows, scorer):
    """Keep checkpoints/templates/decoding/splits separate when aggregating."""
    from collections import defaultdict
    grouped = defaultdict(list)
    dimensions = ('condition', 'checkpoint', 'template', 'decoding', 'split', 'variant', 'task') if scorer == 'choice' else ('condition', 'checkpoint', 'template', 'decoding', 'split', 'variant')
    for row in rows:
        key = tuple((name, row[name]) for name in dimensions if name in row)
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        values = summarize_actions(group) if scorer == 'semantic_action' else summarize_choices(group) if scorer == 'choice' else summarize(group)
        result.append(dict(group=dict(key), metrics=values))
    return dict(scorer=scorer, records=len(rows), groups=result)

def diagnostics(rows, adjacency, distances, requested):
    """Reuse historical cross-template and slot-swap metrics without new calls."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        key = tuple((k, row[k]) for k in ('condition', 'checkpoint', 'decoding', 'split', 'variant') if k in row)
        grouped[key].append(row)
    result = []
    for key, group in sorted(grouped.items()):
        values = {}
        if 'template_consistency' in requested:
            values['template_both_correct'] = template_consistency(group)
        if 'swap' in requested:
            values['swap'] = swap_diagnostics(group, adjacency, distances)
        result.append(dict(group=dict(key), metrics=values))
    return result

def parse_answer(raw, node_count):
    answer = raw.strip()
    format_valid = re.fullmatch(r'(0|[1-9][0-9]*)', answer) is not None
    parsed = int(answer) if format_valid and len(answer) < 20 else None
    valid = parsed is not None and 0 <= parsed < node_count
    return dict(answer=answer, parsed=parsed, format_valid=format_valid, node_id_valid=valid)

def judge(item, raw, adj, distances):
    """Score a report or one-step action using the held-out graph."""
    return score_prediction(parse_output(raw, len(adj)), item, adj, distances)

def score_node(row, adj, distances):
    """Single node correctness/legality implementation for every parser."""
    u, g, task, v = row['u'], row['g'], row['task'], row['parsed']
    valid = row['node_id_valid']
    row['graph_distance'] = int(distances[u, g])
    if task == 'action':
        row['action_legal'] = bool(valid and adj[u, v])
        row['correct'] = bool(row['action_legal'] and distances[v, g] < distances[u, g])
    else:
        row['target'] = u if task == 'report_current' else g
        row['correct'] = bool(valid and v == row['target'])
    return row

def fraction(values):
    values = list(values)
    return {'successes': int(sum(values)), 'total': len(values),
            'rate': float(sum(values) / len(values)) if values else None}

def extract_decision(raw):
    """Extract a declared choice without access to the graph or reference answers."""
    text = raw.strip().replace('**', '').replace('`', '').replace('$', '')
    text = re.sub(r'\\boxed\{([^{}]+)\}', r'\1', text)
    if re.fullmatch(r'(?:node\s+)?(0|[1-9]\d*)[.!。]?', text, flags=re.I):
        return dict(status='identified', source='direct_answer', node=int(re.search(r'\d+', text).group()))
    explicit = []
    for line in text.splitlines():
        match = re.match(r'^\s*(?:#+\s*)?(?:next\s+node|final\s+answer|chosen\s+(?:move|node)|answer)\s*[:=]\s*(?:node\s*)?(\d+)\b(.*)$', line, re.I)
        if match:
            suffix = match[2].strip()
            if re.match(r'^(?:(?:or|and)\b|[,/]|->|→)', suffix, re.I):
                explicit.append(None)
            else:
                explicit.append(int(match[1]))
    if explicit:
        # Multiple distinct explicit decisions are adjudicated, never best-of scored.
        if None not in explicit and len(set(explicit)) == 1:
            return dict(status='identified', source='explicit_decision', node=explicit[-1])
        return dict(status='needs_review', source='conflicting_or_multiple_decisions', node=None)
    statements = []
    for pattern in (
        r'\b(?:I (?:choose|select)|(?:the )?(?:chosen|selected|next) node is|(?:the )?(?:best|correct) (?:next )?(?:move|step) is (?:to )?)\s*(?:node\s*)?(\d+)\b',
        r'\b(?:I (?:will|would) move to|we should move to)\s*(?:node\s*)?(\d+)\b',
    ):
        statements.extend(int(m.group(1)) for m in re.finditer(pattern, text, re.I))
    if statements and len(set(statements)) == 1 and not re.search(r'\b(?:or|and)\s+(?:node\s*)?\d+\b', text, re.I):
        return dict(status='identified', source='choice_in_prose', node=statements[0])
    return dict(status='needs_review', source='no_unique_recognized_decision', node=None)

def score_decision(item, raw, adj, distances, decision=None):
    decision = extract_decision(raw) if decision is None else decision
    node = decision['node']
    scored = judge(item, '' if node is None else str(node), adj, distances)
    scored.pop('format_valid')
    scored.pop('answer')
    scored.update(raw_output=raw, decision=decision)
    return scored

def summarize_actions(rows):
    return dict(total=len(rows), decision_identified=fraction(r['decision']['node'] is not None for r in rows),
                node_id_valid=fraction(r['node_id_valid'] for r in rows),
                action_legal=fraction(r['action_legal'] for r in rows),
                one_step_success=fraction(r['correct'] for r in rows),
                token_limit=fraction(r['hit_token_limit'] for r in rows),
                needs_review=sum(r['decision']['status'] == 'needs_review' for r in rows),
                mean_generated_tokens=sum(r['generated_token_count'] for r in rows) / len(rows),
                success_by_graph_distance={str(d): fraction(r['correct'] for r in rows if r['graph_distance']==d)
                                           for d in sorted({r['graph_distance'] for r in rows})})

def summarize(rows):
    tasks = {task: [r for r in rows if r['task'] == task]
             for task in ('report_current', 'report_goal', 'action')}
    reports = {(r['u'], r['g'], r['task']): r['correct'] for r in rows if r['task'] != 'action'}
    pairs = sorted({(u, g) for u, g, _ in reports})
    both = {(u, g): reports.get((u, g, 'report_current'), False)
            and reports.get((u, g, 'report_goal'), False) for u, g in pairs}
    action = tasks['action']
    return {
        'current_report_exact_accuracy': fraction(r['correct'] for r in tasks['report_current']),
        'goal_report_exact_accuracy': fraction(r['correct'] for r in tasks['report_goal']),
        'both_reports_correct_rate': fraction(both.values()),
        'one_step_success_rate': fraction(r['correct'] for r in action),
        'action_legal_rate': fraction(r['action_legal'] for r in action),
        'format_valid_rate': {t: fraction(r['format_valid'] for r in rs) for t, rs in tasks.items()},
        'node_id_valid_rate': {t: fraction(r['node_id_valid'] for r in rs) for t, rs in tasks.items()},
        'action_success_when_both_reports_correct': fraction(
            r['correct'] for r in action if both.get((r['u'], r['g']), False)),
        'success_by_graph_distance': {str(d): fraction(r['correct'] for r in action if r['graph_distance'] == d)
                                      for d in sorted({r['graph_distance'] for r in action})},
    }

def template_consistency(rows):
    keyed = {(r['template'], r['u'], r['g'], r['task']): r for r in rows}
    result = {}
    for task in ('report_current', 'report_goal', 'action'):
        canonical = [r for r in rows if r['template'] == 'canonical' and r['task'] == task]
        result[task] = fraction(r['correct'] and keyed['heldout', r['u'], r['g'], task]['correct']
                                for r in canonical)
    return result

def swap_diagnostics(rows, adj, distances):
    """Both directions are tested, so the reverse call is the exact slot-swap call."""
    canonical = [r for r in rows if r['template'] == 'canonical']
    keyed = {(r['u'], r['g'], r['task']): r for r in canonical}
    records = []
    for row in canonical:
        reverse = keyed[row['g'], row['u'], row['task']]
        old_score = judge(row, reverse['raw_output'], adj, distances)
        records.append({k: row[k] for k in ('u', 'g', 'task')} | {
            'original_raw_output': row['raw_output'], 'swapped_raw_output': reverse['raw_output'],
            'raw_output_changed': row['raw_output'] != reverse['raw_output'],
            'parsed_output_changed': row['parsed'] != reverse['parsed'],
            'correct_against_original_labels': old_score['correct'],
            'correct_against_swapped_labels': reverse['correct'],
        })
    summary = {t: {name: fraction(r[name] for r in records if r['task'] == t)
                   for name in ('raw_output_changed', 'parsed_output_changed',
                                'correct_against_original_labels', 'correct_against_swapped_labels')}
               for t in ('report_current', 'report_goal', 'action')}
    return {'method': 'reuse_exact_reverse_pair_call', 'summary': summary, 'records': records}

def summarize_choices(rows):
    result=dict(total=len(rows),correct=sum(r['correct'] for r in rows),invalid=sum(not r['valid'] for r in rows))
    result['accuracy']=result['correct']/len(rows)
    groups={label:[r for r in rows if r['answers']==[label]] for label in sorted({r['answers'][0] for r in rows})}
    if rows[0]['task']=='transition_compare':
        result['per_class']={k:dict(total=len(v),correct=sum(x['correct'] for x in v),accuracy=sum(x['correct'] for x in v)/len(v)) for k,v in groups.items()}
        result['macro_accuracy']=float(np.mean([v['accuracy'] for v in result['per_class'].values()]))
        result['confusion']={k:dict(Counter(x['output'] for x in v)) for k,v in groups.items()}
    if rows[0]['task']=='distance_compare':
        anchors={r['anchor'] for r in rows}
        result['swap_consistency']=sum(len({r['output'] for r in rows if r['anchor']==a})==2 for a in anchors)/len(anchors)
    if rows[0]['task']=='neighbor_reason':
        result['analysis_budget_hits']=sum(not r['analysis']['ended'] for r in rows)
    return result
