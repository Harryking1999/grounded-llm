# Rush Hour：任务文字与监督格式

规则对齐[Fogleman公开实现](https://github.com/fogleman/rush/tree/3e3b8396891d1802b30cb3129ae31fdedb319480)；当前使用无墙6×6原题。一次滑动任意合法距离计一个move，不能使用逐格step标签。完整计划已接入；逐轮部分为协议，尚未训练或采用地图。

新训练与评测每题指定一张A已到最右端的完整目标棋盘，所有车的位置均需匹配，字段为`goal_board`。未指定该字段的适配仅为历史题目回放兼容，不安排原规则对照。历史Flash只评过出口目标，不能改称指定终局成绩。

## 共享任务描述与规则

```text
Find a valid sequence of vehicle slides from the initial board to the goal, with as few moves as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board has 6 rows and 6 columns. Each uppercase letter identifies one vehicle. A dot is an empty cell; x is a fixed wall.
Each vehicle has a fixed length and orientation. It can slide only along its orientation, without rotation or sideways movement.
A horizontal vehicle can slide left or right; a vertical vehicle can slide up or down.
An action slides exactly one vehicle by a nonzero integer displacement. Positive means right for a horizontal vehicle or down for a vertical vehicle; negative means left or up.
Every traversed cell must be clear of other vehicles and walls. Vehicles must stay inside the board.
A slide of any legal distance costs exactly one move. Minimize the total number of moves, not the number of cells travelled.
{goal_rule}
Use at most {action_limit} actions.

Vehicle catalogue:
{vehicle_catalogue}

Initial board, top row first:
{initial_board}{goal_board_section}
```

## 地图组／CoT 组：每轮环境更新

两组共享真实状态、指定的完整目标棋盘和合法动作；地图额外提供当前、完整目标状态及合法真实后继的向量。参考成本与解序列不进入文字。

```text
[Environment update]
Current board:
{current_board}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: slide({vehicle}, {displacement}); immediate cost: 1
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects, each with "vehicle" as an uppercase vehicle label and "displacement" as a nonzero integer) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

无工具、无中途反馈，不要求逐步棋盘；完成、最优成本和格式分别计分。

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Sliding vehicle {vehicle} by {displacement} is legal and preserves a minimum-move route to the goal.
<action>{local_id}</action>
```

排序包含最近至多10个候选及`current`，用`<`与`=`；须由实际冻结Q生成，不用精确标签冒充地图。Reason仅在裁判验证后使用；尚未生成SFT数据。无地图组删除排序行。

直接完成：

```text
Reason: This slide reaches the required goal.
<action>{local_id}</action>
```

已经完成：

```text
Reason: The required goal is already reached; no further move is needed.
<done/>
```

## 完整计划结构说明例

仅说明格式，不进入测试题提示。

```json
{"actions": [{"vehicle": "A", "displacement": 4}], "final_status": "solved"}
```
