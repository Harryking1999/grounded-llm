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

Face-grid orientation:
U: the top row touches B; the bottom row touches F; columns run from L to R.
D: the top row touches F; the bottom row touches B; columns run from L to R.
F: the top row touches U; the bottom row touches D; columns run from L to R.
R: the top row touches U; the bottom row touches D; columns run from F to B.
B: the top row touches U; the bottom row touches D; columns run from R to L.
L: the top row touches U; the bottom row touches D; columns run from B to F.

Initial cube state:
{initial_cube_state}

Goal cube state:
{goal_cube_state}
```

格式说明例（复原状态；实际题目由环境生成）：

```text
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
```

采用矩阵便于直接阅读各面；规定行列方向用于消除背面和底面的镜像歧义。贴纸转换实现后须验证其与现有角块状态、六动作转移的一致性。DBL 固定是内部参考系约束，不增加模型动作。

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
