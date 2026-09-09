"""Content-only decoding for Qwen answers before the fixed task judge runs.

The prompts still ask for JSON, but evaluation deliberately treats an
unambiguous JSON array of moves/actions as the same content as its object
wrapper.  We do not infer an answer from an unfinished thinking trace or from
free-form prose: a complete JSON value containing the required action fields
is the minimum evidence used for recovery.
"""
import json


def _strict_object(raw):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _json_values(raw):
    """Return complete JSON values found in left-to-right textual order."""
    decoder = json.JSONDecoder()
    values = []
    for index, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def recover(case, raw):
    """Canonicalize an unambiguous content representation for a task judge.

    Returns ``(output, strict_json, parse_mode)``.  ``output`` is ``None``
    when no complete, task-shaped answer is present.
    """
    strict = _strict_object(raw)
    key = 'actions' if case['condition'].startswith('blocks') else 'path'
    if isinstance(strict.get(key) if strict else None, list):
        return strict, True, 'json_object'

    for value in _json_values(raw):
        if isinstance(value, dict) and isinstance(value.get(key), list):
            return value, False, 'embedded_json_object'
        if not isinstance(value, list):
            continue
        if case['condition'].startswith('blocks'):
            if all(isinstance(action, dict) and all(field in action for field in ('shape_id', 'row', 'col'))
                   for action in value):
                return {'actions': value}, False, 'action_array'
        elif all(isinstance(move, dict) and all(field in move for field in ('from', 'to')) for move in value):
            return {'path': value}, False, 'path_move_array'
        elif len(value) >= 2 and all(type(node) is int for node in value):
            path = [{'from': left, 'to': right} for left, right in zip(value, value[1:])]
            return {'path': path}, False, 'path_node_array'
    return None, False, 'unparseable'


def evaluate(case, raw, tasks):
    """Judge content while retaining format provenance as a diagnostic."""
    output, strict, mode = recover(case, raw)
    if output is None:
        return {'pass': False, 'parseable': False, 'strict_json': False,
                'content_parse_mode': mode, 'failure_type': 'parse_error'}
    verdict = tasks[case['condition']].judge(case, output)
    # ``pass`` comes from the environment judge only.  ``contract_pass`` keeps
    # the old status/state-report fields visible without governing this score.
    return {**verdict, 'parseable': True, 'strict_json': strict,
            'content_parse_mode': mode}
