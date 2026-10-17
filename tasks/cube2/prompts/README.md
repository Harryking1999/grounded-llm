# 二阶魔方：任务文字与监督格式

状态：用户已确定参考 XoT，采用六面 `2×2` 数字矩阵作为 LLM 侧表示。以下共享规则为待确认草案，其余输出合同仍待确认；尚未实现贴纸转换或启动新评测。Q 编码器继续使用角块身份与朝向的 192 维独热输入。

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

六面矩阵与六种四分之一转示例参考 XoT；每个示例独立从同一复原状态出发，不是连续执行。教学示例在各组共用，不提供当前测试题的后继或解题轨迹。已核对这六个示例与当前角块环境的转移一致；完整双向贴纸转换尚未实现。DBL 固定是内部参考系约束，不增加模型动作。

## 地图组／CoT 组：每轮环境更新

待填：真实当前魔方、目标、实际转动历史、候选 ID 与转动的对应关系。地图组额外获得当前／目标／真实后继 Q 槽位。

```text
{environment_update_to_be_discussed}
```

## SOTA 组／32B 组：输出格式

待填完整转动序列、终止状态和是否允许输出额外说明；共享同一规则与呈现。

```text
{sota_32b_output_contract_to_be_discussed}
```

## 监督答案：排序／Reason／Action

待填排序、CoT／短 Reason、单步动作与停止。复原和任意目标条件分别准备；当前复原断点不等于任意目标已解决。
