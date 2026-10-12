from functools import lru_cache

import pytest

from experiments.llm_task_sft.src.random_legal import exact_probabilities


def test_uniform_probabilities_include_losing_actions_and_budget():
    # The middle domino strands two cells; both end-domino choices succeed.
    tiles=[0b0011,0b0110,0b1100]
    empty=exact_probabilities(0b1111,0,32,tiles)
    assert empty['all_legal']==pytest.approx(2/3)
    assert empty['goal_preserving']==pytest.approx(2/3)
    assert empty['first_turn']['reachable_all_legal']==pytest.approx(2/3)
    assert empty['first_turn']['shortest_all_legal']==pytest.approx(2/3)
    goal=exact_probabilities(0b1111,0b0011,32,tiles)
    assert goal['all_legal']==pytest.approx(1/3)
    assert goal['goal_preserving']==1
    assert goal['first_turn']['reachable_all_legal']==pytest.approx(1/3)
    assert goal['first_turn']['reachable_goal_preserving']==1
    assert exact_probabilities(0b1111,0,1,tiles)['all_legal']==0
    assert exact_probabilities(0b0011,0b0011,0,tiles)['all_legal']==1


def test_bitset_dynamic_program_matches_literal_candidate_enumeration():
    tiles=[0b0011,0b0110,0b1100,0b0111,0b1110]
    for goal in range(16):
        for budget in [0,1,2,32]:
            for preserve in [False,True]:
                @lru_cache(None)
                def literal(state,left):
                    if state==goal: return 1.
                    if not left: return 0.
                    candidates=[tile for tile in tiles if tile & state==tile and (not preserve or not tile & goal)]
                    return sum(literal(state ^ tile,left-1) for tile in candidates)/len(candidates) if candidates else 0.
                computed=exact_probabilities(15,goal,budget,tiles)
                assert computed['goal_preserving' if preserve else 'all_legal']==pytest.approx(literal(15,budget))
