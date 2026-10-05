from experiments.flamingo_map_reader.src.analyze_rollout_turns import analyze, summarize


def live(turn, candidates, reachable, kept, **extra):
    return dict(turn=turn, done=False, remaining_shortest=3, candidates=candidates,
                candidate_slots=candidates, reachable_candidates=reachable,
                action_keeps_goal_reachable=kept, legal_action=True, **extra)


def test_live_denominator_keeps_invalid_and_excludes_dead_and_terminal():
    good = live(0, 2, 1, True)
    invalid = dict(live(0, 10, 1, False), legal_action=False)
    dead = dict(turn=1, done=False, remaining_shortest=-1, candidates=1)
    terminal = dict(turn=2, done=True, remaining_shortest=0, candidates=0)
    value = summarize([good, invalid, dead, terminal])
    assert (value['solvable'], value['kept'], value['already_dead']) == (2, 1, 1)
    assert value['rate'] == 0.5
    # Average the per-decision random chance, not total reachable / total slots.
    assert value['random_rate'] == 0.3
    assert value['invalid_or_stop'] == 1


def test_first_dead_end_is_an_executed_legal_removal_not_invalid_output():
    invalid = dict(live(0, 10, 3, False), legal_action=False)
    poison = live(0, 10, 3, False, chosen_action=8)
    dead = dict(turn=1, done=False, remaining_shortest=-1, candidates=4)
    cases = [dict(mode='rollout', turns=[invalid]), dict(mode='rollout', turns=[poison, dead])]
    value = analyze(cases)
    assert value['first_irreversible_dead_end_by_turn'] == {'1': 1}
    assert value['by_turn']['2']['solvable'] == 0
    assert value['by_turn']['2']['rate'] is None
    reference = analyze([dict(mode='reference', turns=[poison])])
    assert reference['first_irreversible_dead_end_by_turn'] == {}
    assert reference['reached'] is None


def test_missing_metric_cannot_silently_disappear_from_live_denominator():
    import pytest
    with pytest.raises(ValueError, match='Replay omitted'):
        summarize([dict(turn=0, done=False, remaining_shortest=2, candidates=5)])


def test_empty_goal_is_separated_from_nonempty_and_zero_move_controls():
    from experiments.flamingo_map_reader.src.analyze_blocks_targets import target_kind
    assert target_kind(dict(start='31', goal='0')) == 'empty'
    assert target_kind(dict(start='31', goal='1')) == 'nonempty'
    assert target_kind(dict(start='31', goal='31')) == 'initial_goal'
    assert target_kind(dict(start='0', goal='0')) == 'initial_goal'
