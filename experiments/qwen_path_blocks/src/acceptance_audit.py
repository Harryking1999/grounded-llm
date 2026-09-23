"""Replay saved answers and diagnose illegal final prefixes before truncation.

This read-only audit does not rescore success, infer plans from exploratory
thinking, or repair incomplete numeric values. It accepts raw run directories
and writes compact evidence, never raw generations, to the requested output.
"""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_eval import evaluate

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/sol_dag_blocks/src'))
from tasks import TASKS


def final_prefix(text, condition):
    """Read consecutive moves from a final JSON array, including a last header.

    A truncated board_after is irrelevant once every action coordinate is
    complete. A scalar at EOF is not complete (e.g. `2` could become `20`).
    Thinking is deliberately not an input to this function.
    """
    key = 'actions' if condition.startswith('blocks') else 'path'
    fields = ('shape_id', 'row', 'col') if key == 'actions' else ('from', 'to')
    match = re.search(r'"' + key + r'"\s*:\s*\[', text)
    if match:
        pos = match.end()
    else:
        stripped = re.sub(r'^\s*```(?:json)?\s*', '', text).lstrip()
        if not stripped.startswith('['):
            return [], 'no_final_array'
        text, pos = stripped, 1
    decoder, prefix = json.JSONDecoder(), []
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text) or text[pos] == ']':
            return prefix, 'complete_elements'
        if text[pos] != '{':
            return prefix, 'unsupported_element'
        try:
            value, end = decoder.raw_decode(text, pos)
        except ValueError:
            # Only inspect flat action fields before the first nested value.
            header = re.split(r'[\[{]', text[pos + 1:], maxsplit=1)[0]
            found = {}
            for field in fields:
                matches = re.findall(r'"' + field + r'"\s*:\s*(-?\d+)(?=\s*[,}])', header)
                if len(matches) != 1:
                    return prefix, 'incomplete_action_header'
                found[field] = int(matches[0])
            prefix.append(found)
            return prefix, 'complete_header_in_truncated_element'
        if not isinstance(value, dict) or not all(type(value.get(f)) is int for f in fields):
            return prefix, 'unsupported_element'
        prefix.append(value)
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos == len(text) or text[pos] == ']':
            return prefix, 'complete_elements'
        if text[pos] != ',':
            return prefix, 'invalid_separator'
        pos += 1


def audit_records(records, cases):
    rows = []
    for record in records:
        case = cases[record['case_id']]
        original = record['verdict']
        truncated = record['response_status'] == 'incomplete'
        prefix, mode = final_prefix(record.get('raw_output', ''), case['condition']) if truncated else ([], None)
        replay = (TASKS[case['condition']].judge(case, {
            'actions' if case['condition'].startswith('blocks') else 'path': prefix}) if prefix else {})
        illegal = replay.get('illegal_action') or replay.get('illegal_move')
        suboptimal = None
        if prefix and not illegal and case['condition'] == 'path_undirected_256':
            distances, queue = {case['goal']: 0}, deque([case['goal']])
            while queue:
                node = queue.popleft()
                for neighbor in case['neighbors'][str(node)]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            for step, move in enumerate(prefix, 1):
                lower_bound = step + distances[move['to']]
                if lower_bound > case['reference']['length']:
                    suboptimal = {'step': step, 'shortest_possible_total': lower_bound,
                                  'reference_length': case['reference']['length']}
                    break
        category = ('illegal_before_truncation' if illegal else 'truncated_unresolved') if truncated else (original.get('failure_type') or 'success')
        if truncated and not illegal and suboptimal:
            category = 'non_shortest_before_truncation'
        rows.append({'case_id': record['case_id'], 'replicate': record['replicate'],
            'condition': case['condition'], 'pass': bool(original['pass']),
            'budget_hit': truncated, 'category': category,
            'prefix_actions': len(prefix), 'prefix_parse_mode': mode,
            'prefix_illegal': illegal,
            'prefix_suboptimal': suboptimal,
            'prefix_state_reports_evaluated': replay.get('state_reports_evaluated', 0),
            'prefix_state_reports_correct': replay.get('state_reports_correct', 0),
            'has_final_text': bool(record.get('raw_output')),
            'has_reasoning_text': bool(record.get('reasoning_text'))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--conditions', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    batches = []
    for directory in args.runs:
        root = Path(directory)
        run = json.loads((root / 'run.json').read_text(encoding='utf-8'))
        suite = json.loads((root / 'suite.json').read_text(encoding='utf-8'))
        cases = {c['id']: c for c in suite['cases']}
        slots = [(r['case_id'], r['replicate']) for r in run['cases']]
        expected = {(c['id'], n) for c in cases.values() for n in range(1, c['replicates'] + 1)}
        if run['status'] != 'completed' or run.get('service_errors') or len(slots) != len(set(slots)) or set(slots) != expected:
            raise ValueError(f'Incomplete or invalid batch: {root}')
        records = [r.copy() for r in run['cases'] if r['condition'] in args.conditions]
        for r in records:
            if r['response_status'] == 'completed':
                r['verdict'] = evaluate(cases[r['case_id']], r['raw_output'], TASKS)
        rows = audit_records(records, cases)
        batches.append({'source': str(root), 'source_commit': run['source_commit'],
            'model': run['api_config']['model'], 'verified_full_slots': len(slots),
            'counts': dict(Counter(r['category'] for r in rows)), 'samples': rows})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'batches': batches}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
