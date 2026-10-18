# 汉诺塔：任务文字与监督格式

状态：整套文本已由用户确认，动作包含盘编号。当前八盘环境、训练与选定Qmap已完成；正式评测的目标限定为指定柱上的整塔，起点可以是任意合法状态。权重与适用范围见[选用记录](../Qmap/selection.json)。

## 共享任务描述与规则

```text
Find a valid sequence of moves from the initial configuration to the goal configuration, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: There are three pegs, A, B, and C, and {disk_count} disks numbered from 1 (smallest) to {disk_count} (largest).
Each peg lists its disks from bottom to top.
An action move(disk_id,from,to) moves the specified disk from one peg to another. The disk must be on top of the source peg.
Only one disk may be moved at a time. Never place a larger disk on a smaller disk.
Each move costs one step.
Reach the goal configuration exactly.

Initial configuration:
{initial_configuration}

Goal configuration:
{goal_configuration}
```

`configuration` 的显示格式如下；数字是固定盘编号，最右侧是柱顶，`[]` 表示空柱。八盘整塔初始摆放的说明例：

```text
A: [8, 7, 6, 5, 4, 3, 2, 1]
B: []
C: []
```

例如执行 `move(1,A,C)` 后：

```text
A: [8, 7, 6, 5, 4, 3, 2]
B: []
C: [1]
```

初始、当前和目标摆放均用相同格式，可表达任意合法目标，不只整塔。盘编号与本轮候选编号相互独立；加入盘编号不改变合法动作集合，解析时须校验它确为源柱顶盘。

## 地图组／CoT 组：每轮环境更新

地图组额外获得当前／目标／合法后继 Q 槽位；两组文字更新相同。

```text
[Environment update]
Current configuration:
{current_configuration}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: move({disk_id},{from},{to})
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current configuration and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with integer "disk_id" and string "from" and "to" fields; peg names are "A", "B", or "C") and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving disk {disk_id} from {from} to {to} is legal and leaves a solvable configuration with one fewer move needed.
<action>{local_id}</action>
```

排序规则沿用积木，最近的至多 10 个候选加 `current`，用 `<`、`=` 表示远近与并列。无地图组直接去掉排序行，保留相同 Reason 与控制标签。

动作直接到达目标时，Reason 改为：

```text
Reason: Moving disk {disk_id} from {from} to {to} is legal and reaches the goal configuration.
```

已在目标时：

```text
Reason: The current configuration exactly matches the goal; no further move is needed.
<done/>
```
