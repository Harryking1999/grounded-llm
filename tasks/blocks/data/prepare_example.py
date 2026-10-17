"""Render a tiny legal fixture through the current SFT prompt and response code."""
import json
from pathlib import Path
from experiments.gcml_counterexamples.src import blocks as rules
from experiments.llm_task_sft.src.protocol import action_key, legal_actions, reason, STOP_REASON
from experiments.map_sft_baselines.src.common import prompt


def main():
    root = Path(__file__).resolve().parents[1]
    # Two horizontal dominoes; keep the lower one as the nonempty goal.
    state = sum(1 << i for i in [0, 1, 33, 34])
    action = dict(shape_id=5, row=0, col=0)
    goal = rules.apply(state, action)
    candidates = legal_actions(state)
    candidate = candidates.index(action) + 1
    assert goal != 0 and goal & state == goal
    assert candidate == sorted(candidates, key=action_key).index(action) + 1
    sample = dict(example_kind="constructed_schema_fixture_not_training_sample", task="blocks",
                  start=str(state), goal=str(goal), board_row=None, shortest_moves=1,
                  route_ids=[candidate], pairs=[[str(state), str(goal)]],
                  initial_board=rules.to_grid(state).split("/"), goal_board=rules.to_grid(goal).split("/"),
                  reference_action=action,
                  sota_answer=dict(actions=[action], final_status="solved"),
                  cot_ra_answer=reason(state, goal, action) + f"\n<action>{candidate}</action>",
                  map_answer_template="{ranking_from_selected_Q}\n" + reason(state, goal, action) + f"\n<action>{candidate}</action>")
    (root / "data/example.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    parts = ["说明：这是构造的单步可解格式例，不是训练集抽样，也不是评测结果。排序须由选定 Q 生成，示例不伪造排序标签。"]
    for title, with_map, order in [("地图组首轮", True, "ranking_first"), ("无地图 R＋A 组首轮", False, "reason_first")]:
        parts.append(f"## {title}\n\n```text\n" + prompt(state, goal, [], first=True, start=state, with_map=with_map, response_order=order) + "\n```")
    parts.append("## 执行后的下一轮更新\n\n```text\n" + prompt(goal, goal, [action]) + "\n```")
    parts.append("## 对应监督答案\n\n```text\n" + sample["cot_ra_answer"] + "\n```\n\n已到达后的停止答案：\n\n```text\n" + STOP_REASON + "\n<done/>\n```")
    (root / "prompts/example.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print("Rendered blocks fixture and both current prompt conditions.")


if __name__ == "__main__":
    main()
