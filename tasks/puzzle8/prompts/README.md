# 8-Puzzle：任务文字与监督格式

状态：参考 [XoT 提示词](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/main/xot_all_in_one/xot/prompter/prompts/prompts_npuzzle.py)的 3×3 数字表示与空格移动方向，按用户要求不加入动作示例，并精简冗余说明。用户已授权使用当前文字开展纯文字基线；目标说明例沿用其左上角为空格的布局，不继承其九步题集／预算。

## 共享任务描述与规则

```text
Find a valid sequence of moves from the initial board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 3x3 grid containing tiles numbered 1 through 8 and one blank, represented by 0.
Each number appears exactly once.
An action Up, Down, Left, or Right moves the blank one cell in that direction by swapping it with the adjacent tile.
Each move costs one step.
Reach the goal board exactly.

Initial board:
{initial_board_3_rows}

Goal board:
{goal_board_3_rows}
```

中文：通过移动空格，将初始棋盘变为目标棋盘；优先完成，再尽量少走。Up／Down／Left／Right 指空格移动方向，数字块相应反向滑入空格。规定方向可避免将“空格向左”误读成“数字块向左”。各组共用上述规则文字，不加入动作教学示例。

目标说明例与 XoT 对齐，初始／当前状态使用同样格式：

```text
0 1 2
3 4 5
6 7 8
```

文字支持显式给定目标；首轮题集先按上述复原目标拟定，任意目标是否纳入正式比较另定。九个数的合法排列不保证与目标连通，准备数据时必须区分合法棋盘与可达任务。默认成功率题集使用可达状态，不把不可达题混入同一分母；若评测无解识别，单独列条件与指标。

## 地图组／CoT 组：每轮环境更新

两组文字相同，规则与目标沿用首轮输入。地图组额外获得当前／目标／全部合法真实后继的 Q 槽位，无地图组不收 Q。

```text
[Environment update]
Current board:
{current_board_3_rows}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: {direction}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

只列空格当前位置允许的 2–4 个动作。候选 ID 是本轮控制地址，不是方向的固定编号；执行后由环境更新真实棋盘。

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current board and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return only JSON with "actions" (an array of strings, each one of "Up", "Down", "Left", or "Right") and "final_status" ("solved", "no_solution", or "unsolved"). Do not include any explanation or commentary outside the JSON.
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

中文：一次输出完整剩余计划，动作使用方向名称，无中途反馈；必须精确达到给定目标。沿用积木的 JSON 与终止语义。模型自报 solved 不替代环境验证。

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving the blank {direction} is legal and leaves a board with one fewer move needed to reach the goal.
<action>{local_id}</action>
```

排序包含本轮全部合法候选与 `current`，按 Q 到目标的距离由近及远排列，用 `<`、`=` 表示远近与并列。动作及“剩余最短步数减少一步”由环境真值另行验证。简短 Reason 中的方向使用小写 up／down／left／right；动作名称保持首字母大写。

直接到达目标时：

```text
Reason: Moving the blank {direction} is legal and reaches the goal board.
<action>{local_id}</action>
```

已在目标时：

```text
Reason: The current board exactly matches the goal; no further move is needed.
<done/>
```

无地图 CoT 组直接删除排序行，保留相同 Reason、动作和停止标签。这里只定义简短 R＋A，不把它称为已准备的长 CoT 数据。
