# 积木提示词对比：当前 1.5B、待定 Astra 与历史基线

本文用于审阅提示词和比较条件。当前 1.5B 与历史模板来自仓库实际代码；Astra 是待确认草案，尚未用于正式评测。研究进度只维护在[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)，本文不维护训练进度或结果表。

当前实验是 **Qwen2.5-1.5B-Instruct + 冻结 Q-map + 可训练地图读取接口**：基座语言模型和地图被冻结，训练读取接口。它逐步接收真实棋盘、合法动作与地图向量。历史 API／Qwen 基线一次输出完整序列，自行推演后续状态。Astra 拟在当前同一批题上做一次性纯文本规划；题目对齐，交互和输入信息不同，因此应单列报告。

## 1. 一张表看区别

| 对比项 | 当前 1.5B + Q-map | Astra 草案（待确认） | 最早 Luna API | 后续 Luna／DeepSeek Flash API | Sol API／Qwen3 4B、8B、32B |
|---|---|---|---|---|---|
| 项目内训练 | 训练读取接口，基座和地图冻结 | 无 | 无 | 无 | 无 |
| 任务目标 | 从起点精确到达给定目标，可保留积木；优先到达，再偏好短解 | 同当前 | 清空棋盘 | 清空棋盘，不要求恢复原分解 | 清空棋盘，不要求最短或恢复原分解 |
| 形状 | 8 种，含二格条 | 同当前 8 种 | 同一套 8 种 | 同一套 8 种 | 10 种：去掉二格条，增加四种 S/Z 形状；编号也改变 |
| 普通题 | 510 题，最短距离 6–9 步 | 复用同一批 510 题 | 8 块构造的清空题 | 8／12 块构造的清空题 | 8／12 块构造的清空题 |
| 终止题 | 已达目标 100 题；固定失败上下文 100 题 | 复用这两类各 100 题 | 未设置对应独立类别 | 未设置对应独立类别 | 未设置对应独立类别 |
| 文本输入 | 初始／目标棋盘；每轮当前棋盘、实际动作历史和全部合法动作 | 当前起点／目标棋盘、实际动作历史 | 初始棋盘、规则 | 初始棋盘、规则 | 初始棋盘、规则 |
| 合法动作候选 | 每轮提供全部候选及临时编号 | 不提供，首步也不提供 | 不提供 | 不提供 | 不提供 |
| 地图信息 | 当前、目标与各合法后继的 Q 向量，经额外模型接口输入 | 不提供 | 无 | 无 | 无 |
| 作答方式 | 每轮选择一步，环境执行后更新 | 一次输出全部剩余动作，无中间反馈 | 一次输出完整动作序列 | 一次输出完整动作序列，无中间反馈 | 一次输出完整动作序列，无中间反馈 |
| 状态由谁维护 | 环境每轮提供真实当前棋盘 | 模型自行推演，裁判提交后回放 | 同 Astra | 同 Astra | 模型自行推演，并输出每步预测棋盘 |
| 输出格式 | 排序文字、候选编号、`<action>`／`<done/>` | 动作 JSON、`solved/no_solution/unsolved`；不要求逐步棋盘 | 动作 JSON，可选短理由 | 动作 JSON，可选短理由 | 动作 JSON，每步必须含 `board_after` |
| 动作预算 | 评测器最多执行 32 步；当前输入文本没有写出这个上限 | prompt 不写动作预算 | 最多 8 步 | 正式 pilot 不限动作数；生成器支持有上限条件 | 不限动作数 |
| 评测重点 | rollout 到达与停止；reference 单步可达、地图排序；两类终止判断 | 拟报告完整到达、动作合法性与两类终止判断 | 合法执行并清空 | 合法执行并清空、失败类型 | 合法执行并清空；预测状态与格式另作诊断 |

### 形状编号不能混用

| 形状 ID | 当前 1.5B／Astra／早期 API | 历史 Sol／Qwen |
|---|---|---|
| 0–3 | `11/10`, `10/11`, `11/01`, `01/11` | 相同 |
| 4 | `1/1` | `1/1/1` |
| 5 | `11` | `111` |
| 6 | `1/1/1` | `011/110` |
| 7 | `111` | `10/11/01` |
| 8–9 | 不存在 | `110/011`, `01/11/10` |

来源：[当前形状](../gcml_counterexamples/configs/pilot.json)、[历史 Sol 形状](../sol_dag_blocks/configs/sol_medium.json)。下文 `{...}` 是动态内容占位符，实际运行会填入完整棋盘等内容，不会把占位符发送给模型。

## 2. 当前 1.5B：实际输入与监督答案

权威入口：[blocks_prompt.py](src/blocks_prompt.py)、[text.py](src/text.py)、[训练合同](configs/blocks_kv_restart.json)、[评测合同](configs/blocks_kv_evaluation.json)。以下展开了正式形状配置，保留动态棋盘占位符。

### 首轮任务文本原文

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
{INITIAL_BOARD_ROWS}

Goal board:
{GOAL_BOARD_ROWS}
```

### 每轮环境更新原文

首轮也追加这一段；之后每次执行模型选择的动作，再发送更新。

```text
[Environment update]
Current board:
{CURRENT_BOARD_ROWS}
Actual executed actions: {EXECUTED_ACTIONS_OR_NONE}
Legal next moves:
{ALL_NUMBERED_LEGAL_MOVES_OR_NONE}
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

候选条目实际写成 `1: remove(shape_id,row,col)` 等；空历史或空候选写成 `(none)`。合法候选包含所有当前允许的移除动作，其中仍可能有会破坏目标可达性的动作。环境不会替模型从候选中选出正确动作。

文本中没有地图距离或正确排序。模型通过分离 K/V 接口读取当前、目标、合法后继的冻结 Q 向量，角色与候选编号用于寻址；这些向量不是新增的一段自然语言 prompt。

### 监督答案原文模板（不是发给模型的额外输入指令）

模型按以下答案接受监督，随后自由生成。来源：[sft.py::decision_text](src/sft.py)。

```text
The current state has not reached the goal.
Map-distance ranking to the goal, closest to farthest: {RANKING}.
Choose candidate {CHOSEN_ID} because its successor has the smallest map distance among the candidates.
<action>{CHOSEN_ID}</action>
```

`{RANKING}` 包含 `current` 与最多 10 个地图最近候选，`<` 表示更近，`=` 表示并列。**环境输入列出全部合法候选，“10”只限制监督答案报告的候选数量。** 数值距离不作为答案监督。

成功终止的监督答案模板：

```text
The current board matches the goal.
Executed actions: {EXECUTED_ACTIONS_OR_NONE}.
Summary: Reached the goal after {MOVE_COUNT} executed removals.
<done/>
```

失败末态的监督答案原文：

```text
No solution: no legal moves remain and the goal has not been reached.
<action>none</action>
<done/>
```

当前无解测试指固定失败历史后的末态：未达目标，且已无合法动作；失败历史中的动作不作为该类 SFT 标签，末态判断才参与监督。它不代表任意复杂无解局面的识别能力。来源：[blocks_failure.py](src/blocks_failure.py)。

### reference 与 rollout 的区别

- **reference**：在固定示范的各个状态上作答；生成本轮答案后，下一轮上下文接入之前的示范答案，而非模型自己的错误动作。它用于测单步选择与排序。
- **rollout**：执行模型自己选择的动作，再提供真实更新；错误会改变后续状态。完整任务到达率来自这一条件。

两者都逐轮生成，不能把 reference 单步准确率当作一次性完整规划成功率。来源：[trajectory_eval.py](src/trajectory_eval.py)。

## 3. Astra：一次性输出完整草案，待确认

这是本文整理的待确认版本：复用当前题集和 8 种形状；不提供首步候选、不输入地图、不要求 `board_after`。只提供一个当前起点棋盘和目标棋盘，删去重复的原始棋盘；规则沿用当前 1.5B 的措辞，输出约定压缩为动作 JSON 与终止状态。**它尚不是已执行的正式配置，也不表示用户已经确认。**

普通题从初始棋盘出发；初始已达目标题应直接停止；失败题从固定历史后的真实末态继续判断。失败题输入仅整理实际棋盘与动作历史，不添加示范排序、地图距离、答案标签或“这题无解”的类别提示。prompt 不写动作预算。

### 拟发送的完整 prompt

```text
Find a valid sequence of removals from the current board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 10x10 binary grid. 1 means occupied and 0 means empty.
An action remove(shape_id,row,col) removes one listed shape at its zero-based top-left anchor.
Every occupied shape cell must overlap a currently occupied board cell and lie inside the board.
Only those cells are removed. Empty shape cells impose no constraint.
Use only the listed orientations. Shapes may be reused; there is no gravity or inventory limit.
Reach the goal board exactly, including any occupied cells that must remain.
Shapes (/ separates rows): 0=11/10; 1=10/11; 2=11/01; 3=01/11; 4=1/1; 5=11; 6=1/1/1; 7=111

Current board:
{CURRENT_BOARD_ROWS}

Goal board:
{GOAL_BOARD_ROWS}

Actual executed actions: {EXECUTED_ACTIONS_OR_NONE}

Plan the complete remaining sequence from the current board and output it once. There is no intermediate feedback; the judge stops at the first illegal action.

Return JSON with "actions" (an array of objects with integer shape_id, row, col) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

### 相对当前 1.5B，改变了什么

| 项目 | 保持一致 | 草案的改变及含义 |
|---|---|---|
| 物理任务 | 同一批普通题与终止题、同样的目标、形状和规则 | 历史清空题不会被混入这 710 题 |
| 动作上限 | 沿用当前评测器上限 | 两者 prompt 均不写动作预算 |
| 棋盘与历史 | 提供真实当前起点、目标及已执行动作 | 不重复原始棋盘；失败历史压成事实文本，不带示范 assistant 答案 |
| 合法动作 | 仍由同一规则裁判验证 | 不列候选，模型自己判断首步及后续合法性 |
| 地图 | — | 不输入 Q-map，测纯文本规划 |
| 交互 | — | 整条序列一次提交，无逐步更新 |
| 输出 | 动作与终止判断 | 用坐标 JSON；不输出候选编号或地图排序 |
| 状态预测 | — | 不要求输出 `board_after`；不因此声称测得逐步状态预测准确率 |

拟优先报告完整到达、完整到达且正确声明成功、非法动作／格式错误，以及两类终止题准确率；不将缺少候选排序的 Astra 答案填入当前 NDCG 列。无解结果仍只解释这 100 个已耗尽合法动作的失败末态。

API 兼容性探测中曾出现请求输出限额未生效；正式预算与实际 token 用量需在正式配置和结果中记录，不能把请求参数当作已约束住的计算预算。本文不保存凭据，也不拟定额外实验调用。

## 4. 历史一次性模板原文

以下都是当时实际模板，不是为本次 Astra 重写的 prompt。Sol 与 Qwen 的任务文字相同；Qwen 外层使用自身 chat template，并开启 thinking。不同模型的推理设置和 token 预算不保证等价。

### 4.1 最早 Luna API：8 种形状，最多 8 步

来源：[run_blocks.mjs::promptFor](../gcml_direct_api_blocks/src/run_blocks.mjs)、[实验说明](../gcml_direct_api_blocks/README.md)。模型记录为 `gpt-5.6-luna`，目标是清空棋盘。

```text
Task: Remove reusable shapes from the following 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left anchor. An action is legal only when every 1-cell of the shape overlaps a current 1; a legal action sets those cells to 0. There is no gravity and no inventory limit. Shapes: 0=11/10, 1=10/11, 2=11/01, 3=01/11, 4=1/1, 5=11, 6=1/1/1, 7=111. Use at most 8 actions.

Initial grid rows:
{GRID_ROWS}

Output: return only valid JSON, with this schema: {"actions":[{"shape_id":0,"row":0,"col":0,"rationale":"short public reason"}],"final_status":"solved"}. Include one object per attempted action in order. The rationale is optional but, if present, must be a short auditable reason. Do not include markdown or hidden chain-of-thought.
```

与当前相比：没有目标棋盘、环境候选或地图；整条序列一次生成；最多 8 步，且不要求报告每步状态。

### 4.2 后续 Luna／DeepSeek Flash API：8 种形状，明确无反馈

来源：[run.py::prompt_for](../gcml_counterexamples/src/run.py)、[Luna 合同](../gcml_counterexamples/configs/pilot.json)、[DeepSeek Flash 合同](../gcml_counterexamples/configs/deepseek_flash.json)。记录的模型为 `gpt-5.6-luna` 与 `deepseek-v4-flash`，共同使用以下任务模板。

```text
Task: Remove reusable shapes from this 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left bounding-box anchor. An action is legal only when every 1-cell of the shape overlaps a current 1; a legal action sets those cells to 0. There is no gravity and no inventory limit. Shapes: 0=11/10, 1=10/11, 2=11/01, 3=01/11, 4=1/1, 5=11, 6=1/1/1, 7=111. {BUDGET_TEXT} Any valid decomposition is accepted; you do not need to recover a particular reference decomposition.

Initial grid rows, from row 0 through row 9 (each character is column 0 through column 9):
{GRID_ROWS}

Plan the complete sequence before answering. Check that the sequence clears the entire grid without an illegal move. The external judge executes the submitted sequence and stops at the first illegal action; there is no intermediate feedback.

Output only valid JSON with an actions array of objects containing integer shape_id, row, and col, and final_status set to "solved" or "unsolved" according to whether your sequence clears the grid. A short public rationale on an action is optional. Do not include markdown or a reasoning transcript.
```

正式 pilot 的 `{BUDGET_TEXT}` 原文为：

```text
There is no action-count limit. A complete decomposition exists. You do not need to minimize the number of actions.
```

生成器也支持有预算条件，替换为：

```text
Use at most {BUDGET} actions. A solution within this budget exists.
```

与最早版相比：更明确坐标、无需恢复参考分解、完整规划和无反馈；正式 pilot 取消了 8 步上限。与当前相比，仍是清空而不是给定目标，不提供候选或地图。

### 4.3 Sol API／Qwen3 对照：10 种形状，每步预测棋盘

来源：[tasks.py::BlocksTask.prompt](../sol_dag_blocks/src/tasks.py)、[Sol 合同](../sol_dag_blocks/configs/sol_medium.json)、[Qwen 调用同一模板](../qwen_path_blocks/src/run.py)、[Qwen 合同](../qwen_path_blocks/configs/thinking.json)。记录的模型为 `gpt-5.6-sol` 和 Qwen3 4B／8B／32B thinking。下文展开正式配置中的形状。

```text
Task: Remove reusable shapes from this 10x10 binary grid until every cell is 0.

Rules: 1 means occupied and 0 means empty. An action is (shape_id, row, col), with zero-based row and column and the shape's top-left bounding-box anchor. Every 1-cell of the selected shape must overlap a CURRENT 1-cell on the board; only those cells become 0. You may NEVER remove an initially empty cell or a cell removed by an earlier action. All shape cells must be within the board. A shape's 0-cells are holes: they do not remove or constrain board cells. There is no gravity, inventory limit, or action-count limit. A complete decomposition exists. Any valid decomposition is accepted; you do not need to minimize actions or recover the original pieces. Only the fixed orientations listed below may be used. In the shape notation, / separates rows.
Shapes: 0=11/10, 1=10/11, 2=11/01, 3=01/11, 4=1/1/1, 5=111, 6=011/110, 7=10/11/01, 8=110/011, 9=01/11/10.

Initial grid rows, from row 0 through row 9 (each character is column 0 through column 9):
{GRID_ROWS}

Plan the COMPLETE sequence before answering. The judge executes it and stops at the first illegal action; there is NO intermediate environment feedback. For EVERY action, predict the entire board immediately AFTER that removal.

Output only valid JSON with an actions array. Each action object must contain integer shape_id, row, col, and board_after, an array of exactly ten strings of exactly ten binary characters (0 or 1), in row order. Also include final_status, "solved" or "unsolved", according to whether the final board is empty. Do not include markdown or a reasoning transcript.
```

`board_after` 是模型预测，不是环境反馈。主任务执行是否成功与状态报告是否满足合同是不同检查。与当前相比：形状集合、目标、交互、输出负担和训练条件都不同，历史 Qwen 清空失败率不能直接作为当前 510 题的基线。

## 5. 本次比较能回答什么

本次 Astra 草案用于测：强模型不读取 Q-map、没有逐步反馈时，能在当前同题测试集上完成多少任务。当前 1.5B 条件用于测：训练后的读取接口能否把地图信息转成逐步动作与终止判断。

二者不能单独证明地图接口的因果收益，因为模型规模、项目内训练、候选信息与交互方式同时变化。若以后要隔离其中某个因素，需要另行设计同模型、同题、同交互的比较；本文仅记录现有条件和待确认草案，不授权新增实验。
