# 推箱子：任务文字与监督格式

推动与行走规则参考 [gym-sokoban](https://github.com/mpSchrader/gym-sokoban/blob/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89/gym_sokoban/envs/sokoban_env.py)，ASCII参考 [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels/blob/78011d8c86ad423bfb2e1005012393b4c3c24409/README.md)。当前小棋盘是诊断条件，不是标准10×10四箱Boxoban题集。

以下定义可复用的文字协议。完整计划的提示与裁判已接入；逐轮地图／CoT部分是协议模板，尚无本任务LLM训练或已采用Qmap权重。

## 共享任务描述与规则

```text
Find a valid sequence of moves that puts every box on a target, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a {board_height}x{board_width} grid with walls, floor, targets, boxes, and one player.
ASCII symbols: # = wall, space = floor, . = target, $ = box, @ = player, * = box on target, + = player on target.
An action U, D, L, or R moves the player one cell up, down, left, or right.
If the adjacent cell contains a box, the action pushes that box one cell in the same direction, only if the cell beyond is empty floor or an unoccupied target.
The player cannot pass through walls or boxes, pull a box, or push more than one box at a time.
Each legal walk or push costs one step. A blocked move is illegal.
The goal is that every box is on a target; the player's final position does not matter.
Use at most {action_limit} actions.

Initial board:
{initial_board}
```

方向指玩家移动方向，行走与推动均为一步。合法但进入死路的动作仍需列为合法候选。目标是状态集合，不显示唯一玩家终点；当前Q输入含目标格。

## 地图组／CoT 组：每轮环境更新

两组共享规则、真实当前状态、实际动作历史与全部合法动作。地图组额外获得当前／目标条件／合法真实后继的Q槽位；精确距离、参考解和死路标签不进入模型文字。

```text
[Environment update]
Current board:
{current_board}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: {direction}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of strings, each one of "U", "D", "L", or "R") and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

沿用既有完整计划条件：无工具和中途反馈，不要求逐步状态报告。模型自报状态不替代动作回放；完成、最优成本与格式分别记录。模型和token预算以实验配置为准。

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving {direction} is legal and leaves a solvable board with one fewer move needed to reach the goal.
<action>{local_id}</action>
```

排序包含最近的至多10个候选及`current`，用`<`、`=`表示远近和并列。排序由实际冻结Q生成，不能把精确距离填入排序冒充Q输出。上述Reason仅在裁判已验证相应事实时作为监督模板；尚未生成本任务SFT数据。无地图CoT组只删除排序行，保留相同Reason和控制标签。

直接到达目标时：

```text
Reason: Moving {direction} is legal and puts every box on a target.
<action>{local_id}</action>
```

已经完成时：

```text
Reason: Every box is on a target; no further move is needed.
<done/>
```

## 完整计划结构说明例

下例对应本任务[data/example.json](../data/example.json)，只说明格式，不作为测试题提示中的few-shot。

```json
{
  "actions": [
    "R"
  ],
  "final_status": "solved"
}
```
