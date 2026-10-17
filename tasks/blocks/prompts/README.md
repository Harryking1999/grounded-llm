# 积木：任务文字与监督格式

状态：现有八形状、10×10 棋盘 SFT 条件已填入。这里的“八形状”是动作目录，不等于每个生成棋盘恰好含八次操作；新的 12 积木条件需另定配置。

## 共享任务描述与规则

以下文字沿用当前 `blocks_prompt.initial_prompt`；初始／目标棋盘由实例填入。

```text
Find a valid sequence of removals from the initial board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 10x10 binary grid. 1 means occupied and 0 means empty.
An action remove(shape_id,row,col) removes one listed shape at its zero-based top-left anchor.
Every occupied shape cell must overlap a currently occupied board cell and lie inside the board.
Only those cells are removed. Empty shape cells impose no constraint.
Use only the listed orientations. Shapes may be reused; there is no gravity or inventory limit.
Reach the goal board exactly, including any occupied cells that must remain.
Shapes (/ separates rows): 0=11/10; 1=10/11; 2=11/01; 3=01/11; 4=1/1; 5=11; 6=1/1/1; 7=111

Initial board:
{initial_board_10_rows}

Goal board:
{goal_board_10_rows}
```

## 地图组／CoT 组：每轮环境更新

```text
[Environment update]
Current board:
{current_board_10_rows}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: remove({shape_id},{row},{col})
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

使用真实执行结果更新，不能把模型自报棋盘当真值。本轮 ID 不是全局动作编号。地图组另收当前／目标／全部合法后继的冻结 Q 槽位；无地图组不收 Q。

## 输出格式与 CoT 监督答案

当前地图组使用 ranking-first：先输出地图距离排序，再输出简短 Reason，最后输出一个控制标签。

```text
Map-distance ranking to the goal, closest to farthest: {nearest_10_candidates_plus_current_with_<and_=}.
Reason: Clearing {cells} is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
<action>{local_candidate_id}</action>
```

若本次动作直接到达目标，Reason 为 `Reason: Clearing {cells} is legal and leaves exactly the goal board.`；已在目标时为：

```text
Reason: The current board exactly matches the goal; no further removal is needed.
<done/>
```

当前无地图 7B 监督为相同事实依据与动作，即 R→A，不输出排序。它使用简短 Reason，并非长 CoT。后续长 CoT 文本、是否改为排序＋A，均待讨论，未替换已有运行。排序是 Q 距离顺序，动作正确性由环境裁判另判。

完整实际首轮格式指令与后续更新见同目录 `example.md`，由 `python -m tasks.blocks.data.prepare_example` 调用当前运行的提示函数生成。权威实现：`experiments/flamingo_map_reader/src/blocks_prompt.py`、`experiments/map_sft_baselines/src/common.py`、`experiments/llm_task_sft/src/protocol.py`。

## SOTA 组／32B 组

采用上面的共享任务描述与规则。按用户确定的完整计划条件，一次输出剩余动作，无中途环境反馈：

```text
Plan the complete remaining sequence from the current board and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with integer shape_id, row, col) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

动作使用物理参数而非本轮候选 ID；不要求 `board_after`。已在目标时使用 `{"actions":[],"final_status":"solved"}`。输出预算待正式配置确认。与地图／CoT 逐轮条件的反馈方式不同，报告时注明。旧 SOTA／32B 整段计划条件还使用过不同形状目录与空目标，不能直接移用为本轮匹配结果。
