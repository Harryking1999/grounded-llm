"""Aggregate saved decisions without importing models or GPU libraries."""

def summarize(cases):
    turns = [row for case in cases for row in case['turns']]
    live = [row for row in turns if row['live']]
    ranked = [row for row in turns if row['rank_eligible']]

    def rate(key, rows):
        numerator = sum(int(row.get(key, False)) for row in rows)
        return dict(numerator=numerator, denominator=len(rows),
                    value=numerator / len(rows) if rows else None)

    result = dict(tasks=len(cases), arrival=sum(case['arrived'] for case in cases),
                  arrival_and_stop=sum(case['arrived_and_stopped'] for case in cases),
                  safe=rate('safe', live), optimal=rate('optimal', live),
                  map_optimal=rate('map_optimal', ranked))
    for key in ('ndcg_inverse_rank_at_1', 'ndcg_inverse_rank_at_10',
                'ndcg_minmax_distance_at_1', 'ndcg_minmax_distance_at_10'):
        rows = [row for row in ranked if key in row]
        result[key] = dict(denominator=len(rows),
                           value=sum(row[key] for row in rows) / len(rows) if rows else None)
    return result
