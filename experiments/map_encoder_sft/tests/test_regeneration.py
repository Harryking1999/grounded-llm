"""New-map actions replace old route labels while teacher certification stays on."""
from pathlib import Path

from experiments.map_sft_baselines.src.generalization_prepare import certify
from grounded_llm.tasks.blocks import rules


ROOT = Path(__file__).resolve().parents[3]


def test_new_map_replans_and_certifies_instead_of_reusing_old_actions():
    tile = rules.PLACEMENTS[0][0]
    row = dict(task='blocks',variant='partial8',start=str(tile),goal='0',
        map_checkpoint=str(ROOT/'tasks/blocks/Qmap/checkpoints/decision_clear_300_4f413b5.pt'),
        context=dict(coordinate_frame='goal_xor'),reference_actions=['obsolete_action'])
    accepted,steps = certify(row,ROOT)
    assert accepted['reference_actions'] != row['reference_actions']
    assert len(accepted['reference_actions']) == accepted['reference_cost'] == 1
    assert steps[-1].done
    for step,action in zip(steps,accepted['reference_actions']):
        chosen=step.candidate_actions.index(action)+1
        assert chosen in step.map_minimal_candidates
        assert step.execute(chosen)[1] == 0
