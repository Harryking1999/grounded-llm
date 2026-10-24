# 二阶魔方：任务文字与监督格式

状态：用户已确定参考 XoT，采用六面 `2×2` 数字矩阵和动作前后示例作为 LLM 侧表示。共享规则、每轮更新、完整计划输出和监督答案已补齐，用户已授权使用当前文字开展纯文字基线。双向贴纸转换已实现并检查；Q 编码器继续使用角块身份与朝向的 192 维独热输入。

## 共享任务描述与规则

表示参考 [XoT 公开提示词](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/main/xot_all_in_one/xot/prompter/prompts/prompts_cube.py)：六面分别列出两行数字，0–5 是颜色类别。采用其一种复原配色：U=0、R=1、F=2、D=3、L=4、B=5。初始、当前和目标状态统一使用此格式。

当前地图使用 DBL 角块固定参考系，动作保持 U、U′、R、R′、F、F′ 六种四分之一转；不引入 XoT 的半转单步或 11 步上限。目标沿用本项目的明确目标状态。

```text
Find a valid sequence of turns from the initial cube state to the goal cube state, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: This is a 2x2 cube with six faces: Upper (U), Front (F), Down (D), Left (L), Right (R), and Back (B).
Each face is shown as a 2x2 grid. Numbers 0 through 5 represent fixed sticker colors.
The cube's reference frame stays fixed; do not rotate the whole cube.
Allowed turns are U, U', R, R', F, and F'. A letter turns that face 90 degrees clockwise as viewed directly from outside that face; an apostrophe means counterclockwise.
Each quarter-turn costs one step. A 180-degree turn requires two quarter-turns.
Reach the goal cube state exactly.

Action examples: each example starts independently from the state below.
Do not apply these example turns in sequence.

[Before each example]
Upper:
0 0
0 0
Front:
2 2
2 2
Down:
3 3
3 3
Left:
4 4
4 4
Right:
1 1
1 1
Back:
5 5
5 5

[After U]
Upper:
0 0
0 0
Front:
1 1
2 2
Down:
3 3
3 3
Left:
2 2
4 4
Right:
5 5
1 1
Back:
4 4
5 5

[After U']
Upper:
0 0
0 0
Front:
4 4
2 2
Down:
3 3
3 3
Left:
5 5
4 4
Right:
2 2
1 1
Back:
1 1
5 5

[After R]
Upper:
0 2
0 2
Front:
2 3
2 3
Down:
3 5
3 5
Left:
4 4
4 4
Right:
1 1
1 1
Back:
0 5
0 5

[After R']
Upper:
0 5
0 5
Front:
2 0
2 0
Down:
3 2
3 2
Left:
4 4
4 4
Right:
1 1
1 1
Back:
3 5
3 5

[After F]
Upper:
0 0
4 4
Front:
2 2
2 2
Down:
1 1
3 3
Left:
4 3
4 3
Right:
0 1
0 1
Back:
5 5
5 5

[After F']
Upper:
0 0
1 1
Front:
2 2
2 2
Down:
4 4
3 3
Left:
4 0
4 0
Right:
3 1
3 1
Back:
5 5
5 5

Initial cube state:
{initial_cube_state}

Goal cube state:
{goal_cube_state}
```

六面矩阵与六种四分之一转示例参考 XoT；每个示例独立从同一复原状态出发，不是连续执行。教学示例在各组共用，不提供当前测试题的后继或解题轨迹。已核对这六个示例与当前角块环境的转移一致；[双向贴纸转换](../../../experiments/task_text_baselines/src/cube2.py)已通过随机状态往返检查。DBL 固定是内部参考系约束，不增加模型动作。

## 地图组／CoT 组：每轮环境更新

两组使用相同文字，初始规则与目标沿用首轮输入。当前状态使用上面的六面矩阵格式，转动历史来自环境实际执行结果。地图组额外获得当前／目标／六个真实后继 Q 槽位；无地图组不收 Q。

```text
[Environment update]
Current cube state:
{current_cube_state}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: {turn}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

`{turn}` 为 U、U'、R、R'、F、F' 之一。每轮列出六个候选；候选 ID 是本轮控制地址，不是固定的转动编号。执行后由环境更新真实状态，不采用模型自报的状态。

## SOTA 组／32B 组：输出格式

共享同一任务规则、六面表示和固定动作示例。一次输出完整剩余计划，无中途反馈；动作使用转动名称，不使用本轮候选 ID。

```text
Plan the complete remaining sequence from the current cube state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return only JSON with "actions" (an array of strings, each one of "U", "U'", "R", "R'", "F", or "F'") and "final_status" ("solved", "no_solution", or "unsolved"). Do not include any explanation or commentary outside the JSON.
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

格式示例（不对应上面的具体魔方）：

```json
{"actions":["R","U'","F"],"final_status":"solved"}
```

已在目标时输出 `{"actions":[],"final_status":"solved"}`。是否到达由环境执行验证，此组与地图／CoT 的逐轮反馈条件不同，结果中注明。

## 监督答案：排序／Reason／Action

地图组沿用积木的排序→简短 Reason→单步动作：

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Turning {turn} is legal and leaves a cube state with one fewer turn needed to reach the goal.
<action>{local_id}</action>
```

排序包含六个候选和 `current`，按 Q 到目标的距离由近及远排列，用 `<`、`=` 表示远近与并列。动作正确性及“剩余最短步数减少一步”的依据由环境真值另行验证。

动作直接到达目标时：

```text
Reason: Turning {turn} is legal and reaches the goal cube state.
<action>{local_id}</action>
```

已在目标时：

```text
Reason: The current cube state exactly matches the goal; no further turn is needed.
<done/>
```

无地图 CoT 组直接去掉排序行，保留相同 Reason、动作和停止标签。以上均为简短 R＋A 监督，复原和任意可达目标共用文字格式，但数据与结果分别报告。
