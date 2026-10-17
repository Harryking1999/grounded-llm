"""Apply an output-token scoring limit without changing saved execution evidence."""
import json
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1]/'configs/budget_16k.json'


def output_limit(path=POLICY):
    limit = json.loads(Path(path).read_text(encoding='utf-8'))['max_output_tokens']
    if type(limit) is not int or limit <= 0:
        raise ValueError('Output budget must be a positive integer')
    return limit


def rescore(record, limit):
    result = dict(record)
    execution = record.get('execution_verdict',record['verdict'])
    result['execution_verdict'] = execution
    result['verdict'] = dict(execution)
    result['scoring_output_limit'] = limit
    if record.get('service_error'):
        result['budget_status'] = 'service_error'
        return result
    tokens = (record.get('usage') or {}).get('output_tokens')
    status = ('unknown' if type(tokens) is not int or tokens < 0 else
              'exceeded' if tokens > limit else 'within')
    result['budget_status'] = status
    if status != 'within':
        result['verdict'].update(solved=False,contract_pass=False,
            optimal=False if execution.get('optimal') is not None else None,
            failure='budget_exceeded' if status == 'exceeded' else 'budget_unknown')
    return result
