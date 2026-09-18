"""External graph judge; generated answers never receive reference distances."""

import re


def parse_answer(raw, node_count):
    answer = raw.strip()
    format_valid = re.fullmatch(r'(0|[1-9][0-9]*)', answer) is not None
    parsed = int(answer) if format_valid and len(answer) < 20 else None
    valid = parsed is not None and 0 <= parsed < node_count
    return dict(answer=answer, parsed=parsed, format_valid=format_valid, node_id_valid=valid)


def judge(item, raw, adj, distances):
    """Score a report or one-step action using the held-out graph."""
    row = {**item, 'raw_output': raw, **parse_answer(raw, len(adj))}
    u, g, task, v = item['u'], item['g'], item['task'], row['parsed']
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
