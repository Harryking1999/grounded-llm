# 积木 1.5B 无地图动作 SFT 基线方案

首轮用 **Qwen2.5-1.5B-Instruct 做全参数 SFT**，检验任务微调能否提升积木求解表现。

- **模型输入：**规则、目标棋盘、当前棋盘、已执行动作、全部合法候选；不提供认知地图。
- **模型输出：**每轮选一个动作，环境执行后给出新状态；到达目标时输出停止标记。
- **监督来源：**每题一条经求解器认证的合法最短轨迹，拆成多轮动作标签；不生成排序或 CoT。

## 1. 实验怎样运行

```mermaid
flowchart LR
    O["离线最短轨迹"] --> S["全参数 SFT"] --> L["1.5B LLM"]
    E["环境：当前棋盘＋合法候选"] --> L
    L -->|"一个动作／停止"| E
```

**一次选一个动作，整题通过多轮交互完成。** 环境负责执行、更新棋盘和列出合法候选；模型根据新状态选择下一步。这里的“动作序列”指各轮输出连成的序列。

求解器只在离线制作标签、认证最短解和评测时使用。运行中的模型不调用求解器，也不会收到最短距离、推荐动作或后继可达性。

| 对照条件 | 训练方式 | 每轮信息 |
|---|---|---|
| 无地图动作 SFT（本方案） | 全参数微调 1.5B | 棋盘、目标、历史、全部合法候选 |
| 未微调同基座 | 不训练 | 与 SFT 相同，使用相同输出格式 |
| 现有地图接口系统 | 冻结 LLM／地图，训练接口 | 上述信息＋认知地图 |

地图接口沿用原答案格式，对照时提取实际动作和终止结果。

SFT 与未微调同基座的差异用于衡量任务微调收益。与地图接口的比较属于系统比较：训练范围和监督也不同，不能把全部收益单独归因于地图。

## 2. 模型实际收到什么、输出什么

### 2.1 输入和输出约定

沿用[现有积木题面与环境更新格式](../experiments/flamingo_map_reader/src/blocks_prompt.py)：

1. **首轮：**任务规则、八种形状、初始棋盘、目标棋盘、输出格式，以及当前环境状态。
2. **后续轮：**完整 10×10 当前棋盘、实际执行历史、全部合法候选；保留前面各轮的对话。
3. **assistant：**只输出 `<action>ID</action>` 或 `<done/>`，这些响应就是训练标签。

候选编号只在当前轮有效；它对应真实的 `remove(shape_id,row,col)` 动作。合法候选只表示几何上能执行，仍可能进入死局，因此必须全部呈现，不能用求解器预先过滤。输入不包含地图向量、K/V、距离、排名或最优标记。

**停止条件是当前棋盘精确等于目标。** 初始已达目标的题直接输出 `<done/>`；即使目标非空且还有合法移除动作，也应停止。

### 2.2 完整三轮示例

这是一个用于说明格式的两步小题，不代表正式训练题的难度或长度。初始棋盘的第一行有连续四格，右下角一格必须保留。选择两次水平二格移除，最后停止。

下面每个 `user` 文本框都表示一条完整输入消息；`assistant` 文本框是该轮的监督答案。英文题面沿用现有实验。

#### 第 1 轮：读取题目，选择第一次移除

**user 输入**

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
1111000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000001

Goal board:
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000001

Response format:
If the current board exactly matches the goal, output only <done/>.
Otherwise, choose one legal candidate from the current update and output only <action>ID</action>.

[Environment update]
Current board:
1111000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000001
Actual executed actions: (none)
Legal next moves:
1: remove(5,0,0)
2: remove(5,0,1)
3: remove(5,0,2)
4: remove(7,0,0)
5: remove(7,0,1)
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

**assistant 监督答案**

```text
<action>1</action>
```

环境执行 `remove(5,0,0)`，移除 `(0,0)`、`(0,1)`，再发送更新后的棋盘。

示例说明：首轮候选 1 和 3 都能在两步内完成；候选 2、4、5 虽然合法，却会留下无法继续移除的孤立格。本例固定选候选 1。这段说明用于读者理解，不写入模型输入或监督答案。

#### 第 2 轮：根据新棋盘选择第二次移除

**user 输入**（前面的题目和对话仍保留在上下文中）

```text
[Environment update]
Current board:
0011000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000001
Actual executed actions: remove(5,0,0)
Legal next moves:
1: remove(5,0,2)
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

**assistant 监督答案**

```text
<action>1</action>
```

这次编号 1 对应 `remove(5,0,2)`，移除 `(0,2)`、`(0,3)`。两轮编号相同，实际动作不同。

#### 第 3 轮：确认到达目标并停止

**user 输入**

```text
[Environment update]
Current board:
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000001
Actual executed actions: remove(5,0,0) -> remove(5,0,2)
Legal next moves:
(none)
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

**assistant 监督答案**

```text
<done/>
```

本题的完整训练样本包含上述三轮对话。只对两次动作响应、停止响应及各自的 assistant 结束 token 计算损失；题面、棋盘、候选和 padding 不计算损失。

## 3. 监督数据怎样制作

### 3.1 使用哪些题

复用[当前积木合同](../experiments/flamingo_map_reader/configs/blocks_kv_restart.json)对应的物理题和固定划分，重新制作动作标签。

| 来源 | 本轮处理 |
|---|---|
| 8,000 普通训练题 | 每题一条合法最短轨迹 |
| 1,000 初始已达目标题 | 每题一轮 `<done/>` |
| 1,000 固定失败历史 | 不加入本轮最短解监督 |
| 普通 validation／test | 保持原划分；普通 test 为 510 题，保留原初始棋盘新任务／新初始棋盘分组 |
| 已达标 100 题 | 源合同属于 validation，单列停止评测 |

实施前核对实际清单和正式源路径，不能用历史 preview 替代。原题经过地图贪心成功筛选，这一选题偏差需披露；本轮只换标签，不重新选题。

### 3.2 每题如何得到一条标签轨迹

1. **求最短解。**确认目标格均属于初始占据格，对“初始减目标”调用[精确覆盖求解器](../experiments/gcml_counterexamples/src/blocks.py)，得到最少移除次数和一条解。多个最短解按固定动作遍历顺序取一条，不采样多条路线。
2. **回放验证。**按真实规则执行这条解：每步合法、目标格始终保留、终局精确等于目标，动作数等于认证最短值。
3. **生成对话。**每步重新枚举全部合法候选，将解中的物理动作映射为本轮编号，填写棋盘和实际历史，生成 `<action>ID</action>`；终局追加 `<done/>`。
4. **核对距离。**用[精确距离裁判](../experiments/blocks_distance_map/src/oracle.py)检查每个动作让剩余最短距离减少 1。求解超时记为 unknown，查明后再制作正式数据，不能当无解或静默删题。

最短性来自精确求解和验证，不来自地图贪心成功。直接动作监督不需要教师生成解释。

### 3.3 编号增强和数据隔离

- **编号增强：**沿用现有策略，普通题最多 6 个版本、零步题 1 个版本。每个版本只改变候选编号，物理轨迹相同，答案编号同步变化。
- **划分隔离：**同题所有版本留在同一 split。检查新最短轨迹的非终止 `(状态,目标)` 是否与留出输入重合，固定测试题并更新实际曝光分组；共有终局不等于整题泄漏，零步完全同题重合单列。
- **保存内容：**题目来源、物理动作、状态、候选编号映射、最短认证和种子。生成数据与缓存放 Git 外。

## 4. 怎样训练

### 4.1 已定训练起点

| 项目 | 设置 |
|---|---|
| 基座／更新范围 | 原始 Qwen2.5-1.5B-Instruct，全参数，不使用地图接口或 LoRA |
| 学习率／有效 batch | 2e-5／32 条完整多轮轨迹 |
| 训练预算 | 3 epoch；每 epoch 遍历全部编号版本，固定一个训练种子 |
| 学习率调度 | 前 5% 更新 warmup，之后 cosine 至峰值 1/10 |
| 优化器 | AdamW，weight decay 0.01，梯度裁剪 1.0 |
| 工程设置 | 设备支持时 BF16、梯度检查点；不跨题 packing |

单卡 batch 和梯度累积按最长样本实测。沿用已选的超参，先完成 1.5B 基线。

**每个物理题保持相同训练权重：**先平均一条轨迹内的监督 token 损失，再平均同题编号版本，最后平均物理题。这样，编号版本多的普通题不会挤低零步题权重。

实现时，若 N 个物理题扩展为 M 条轨迹、题 i 有 n_i 个版本，对轨迹平均损失乘固定权重 `M/(N*n_i)`，再做 batch 平均；不按 batch 内权重和再次归一化。

### 4.2 长度和交互预算

| 上限 | 如何确定 |
|---|---|
| 训练序列长度 S | 用真实 tokenizer／chat template 统计完整多轮训练样本，以最大长度确定 |
| 每轮生成长度 G | 统计动作编号响应和停止响应的实际 token 长度，包含结束 token；用 validation 检查触顶后固定 |
| 执行动作数 | 最多 32 个动作，另允许到达后的停止轮 |

S 不能裁掉后续动作或停止标签；推理时累计上下文＋G 也须在原生上下文内。若超限，实施前处理并记录，不能静默截断或删题。未微调与 SFT 使用相同 G，不用 test 调预算。

### 4.3 现有实现怎样复用

- **可复用：**环境规则、候选编号、多轮消息拼接，以及将 user token 设为 -100 的思路。
- **需要改：**普通 causal LM 入口、全参数更新、完整模型保存与重载、纯动作／停止解析和评分；移除地图加载与 token-map 分配。

旧 `MapReader` 冻结基座，旧 trainer 只保存 `adapter.pt`，旧评分依赖排序和自然语言总结。因此，正式实施时需要调整训练和评测入口，不能只改提示词。

## 5. 怎样评测

### 5.1 两种评测方式

**整题交互（rollout）：**从固定初始棋盘出发，执行模型自己选的动作，回报真实棋盘，直到停止、失败或用完 32 个动作。保留模型自身回答，衡量实际解题能力。

**固定状态单步评测（reference）：**给模型固定最短参考轨迹中的历史和当前棋盘，让它选择一次动作。下一轮接入正确参考动作，避免本轮错误改变后续输入；当前答案不能提前放进输入。

两种方式都每轮只做一次贪心生成，不重采、不修复动作、不提供求解器建议。

### 5.2 主要指标

| 指标 | 计分方式 |
|---|---|
| 轨迹到达率（主指标） | 32 动作内曾精确到达目标的普通题数 ÷ 全部普通题数 |
| 到达且正确停止率 | 到达后正确输出 `<done/>` 的普通题数 ÷ 全部普通题数 |
| 单步可达率 | 固定状态评测中，合法选择且后继仍可解的轮数 ÷ 作答前可解的非终止轮数 |
| 辅助指标 | 单步最优率、非法／格式错误、提前停止、生成触顶及 token 数；零步停止正确率单列 |

单步最优指动作令剩余最短距离减少 1。多个动作可能同样最优，按真实状态和精确距离评分，不要求与唯一训练标签编号一致。

单步可达与合法不同：几何合法仍可能破坏可解性。提前停止、编号越界和格式错误在原本可解的决策轮计失败。整题交互中的单步可达率另列，并记录已死局轮数。

失败处理明确如下：

- 未到目标且无合法候选：结束并记失败。
- 非法动作或格式错误：不执行，结束并保留失败计数。
- 到达目标：再给一个停止轮；即使未正确停止，也分别记录“已到达”和“停止失败”。

### 5.3 选模和对照

每 epoch 末评固定 validation。按普通轨迹到达率选 checkpoint；平分时看到达且正确停止率，再平分选较早 checkpoint。test 只评未微调基座与选定的 SFT checkpoint。

与地图接口比较时，题目、候选编号策略和动作预算一致。固定状态评测还须使用相同状态及历史，不能直接拿不同参考轨迹的旧均值作对照。本基线不评价排序、地图最优率或 NDCG。

## 6. 实施顺序与完成条件

1. **核对源数据。**固定题目清单、split、来源和编号策略。
2. **制作 32 题试样。**固定 32 道训练题，完成最短认证、状态回放、编号映射和停止标签；记录 unknown 与求解成本。
3. **实现训练和评测入口。**确认 mask、轨迹权重、参数更新和完整模型重载正确；小量题能过拟合，评分不依赖排序文本。
4. **制作全量标签，测长度和资源。**确定 S、G、显存、吞吐及 batch／累积设置；明确处理未解决题和曝光冲突。
5. **提交正式配置后运行。**训练 3 epoch，validation 选模，固定 test 保存原始回答与逐步执行结果。

正式实施时在 `experiments/llm_task_sft/` 建立配置和必要实现；配置形成后以其为参数权威。本次仅更新方案。

7B、寻路、教师解释和 CoT SFT 留待后续讨论。
