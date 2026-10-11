# 1.5B 整题 CoT SFT 基线方案

**主流程：固定物理题 → 求解器生成多条合法计划 → 原始 1.5B 根据计划和状态证据写解释 → 验收 → 全参数 SFT → 无工具整题评测。** 寻路、积木分别训练。本文件是实施方案，当前进度见[项目状态页](PROJECT_STATUS_AND_TODO.md)。

## 1. 实验要回答什么

检验带依据的整题监督能否提高 1.5B 的任务完成率。首轮设置如下：

| 条件 | 参数更新 | 模型收到什么、怎样作答 |
|---|---|---|
| 整题 CoT SFT | Qwen2.5-1.5B-Instruct 全参数 | 规则＋完整题目；一次输出解释与完整计划 |
| 未微调同基座 | 无 | 与 SFT 相同的题面、格式和生成预算 |
| 已有地图接口系统 | 只训练接口，LLM／地图冻结 | 地图、逐步真实状态与合法候选；作为已有系统参照 |

前两项衡量 SFT 收益；第三项同时改变信息和交互条件，只作系统级比较。首轮结果不能单独证明地图的因果收益，也不能分离 CoT 与计划监督的贡献。

**监督选择已确认：**使用未做本项目微调的原始 1.5B 作为解释教师；程序负责找路线，教师负责解释路线。保留有依据的局部比较与排除，不要求完整搜索日志。这能监督约束判断、状态推演和计划验证；能否改善自主找解由整题评测回答。

## 2. 怎样制作训练数据

### 2.1 固定题目与划分

复用现有实验保存的物理题，保留来源、task ID、状态、目标、split 和组别；重新生成监督，不重新筛题。

| 任务 | 来源合同 | 训练题／留出题边界 |
|---|---|---|
| 寻路 | [path_single_long.json](../experiments/flamingo_map_reader/configs/path_single_long.json) | 4,000 普通＋192 零步训练题；同一 256 节点图，目标按 192／32／32 划分；普通 validation／test 为 500／1,000 题 |
| 积木 | [blocks_kv_restart.json](../experiments/flamingo_map_reader/configs/blocks_kv_restart.json) | 8,000 普通＋1,000 零步训练题；排除 1,000 条固定失败历史 |
| 积木评测 | [blocks_kv_evaluation.json](../experiments/flamingo_map_reader/configs/blocks_kv_evaluation.json) | 普通 test 510 题；分别报告原初始棋盘新任务与新初始棋盘 |

上述是合同预期，**实施第一步要找到并核对实际保存清单**。本地旧 `path10_manifest.json` 与 `blocks_long_manifest_preview.json` 不能替代当前来源。原训练题按地图贪心成功筛选的偏差保留并披露；失败历史不等于原整题无解。旧积木 100 道已达标题属于 validation，独立零步 test 的来源需另行确认。

学生题面包含真实规则及完整物理输入：寻路给完整邻接表、起点和目标；积木给八种形状、10×10 初始／目标棋盘及移除规则。一条计划配一份题面，不复制旧候选编号增强；输入呈现按保存种子固定。

### 2.2 每题取得哪些计划

- 普通题最多 **4 条不同计划**：至少一条最短，其余不超过最短＋2步。不足 4 条保留实际数量，不复制补足或删题。
- 寻路用 BFS 认证最短距离，再在该距离至＋2步内枚举不重复节点的路径，用到目标的距离下界剪枝。visited 只限制当前路径，不能用全局 visited 丢掉其他路线。
- 积木先确认目标是初始占据格的子集，对待移除格求最小精确覆盖，再有界枚举补充覆盖：选候选最少的剩余格，遍历覆盖它的放置并递归余格，用 `ceil(剩余格数/3)` 作动作数下界剪枝。当前规则无重力／库存限制，合法覆盖互不重叠，动作顺序可交换；优先不同覆盖，不足时可用换序，并分别统计两种多样性。
- 零步只有一份短监督：寻路 `path=[start], steps=0`；积木 `actions=[], steps=0`，解释初始已等于目标。

复用[图距离实现](../experiments/flamingo_map_reader/src/data.py)与[积木覆盖求解器](../experiments/gcml_counterexamples/src/blocks.py)，只读取真实图／规则，不调用地图评分。最短认证和补充枚举分别设预算，取够即停；预算由试样耗时确定。超时记 unknown，不能记无解；最短未认证的题保留在问题清单中，先解决再进入正式语料。

同题多解始终属于同一 split。检查新监督中的非终止 `(状态,目标)` 是否撞到留出输入；优先重采训练路线，无法消除则记录。测试题保持固定，按新增状态重算曝光分组；共有终局不等于整题泄漏，零步同题重合需单列。

### 2.3 让原模型为每条计划写解释

| 输入／输出 | 内容 |
|---|---|
| 教师输入 | 学生题面＋一条指定计划＋独立回放的动作效果、后继状态和目标保持证据 |
| 教师输出 | `<reasoning>` 中写目标约束、方案依据、逐步状态和终局检查；`<final>` 中写指定计划的 JSON |
| 学生输入 | 只有题面和输出格式要求 |
| 学生监督 | 验收后的完整解释＋计划；不含教师的路线提示或事实包 |

解释必须独立可读。多解均可行时不编造排他理由；没有失败分支证据时不编造探索／回溯；声称最短时给出可检查的下界或证明。两个具体输入与监督输出见附录 A。零步用规则短文本即可。

**建议试样协议：**每任务固定 32 道训练题，覆盖距离、棋盘来源、非空目标与候选较多的题，每条所选计划生成一次解释。教师贪心生成；按原生上下文减实际输入长度确定可用生成额度，记录触顶。失败最多按具体错误反馈修订一次。小批全部人工审阅，记录首次／修订后验收率、错误类型、最长文本、token 和耗时。教师与学生初始权重相同，新增信息来自离线计划与事实；解释质量尚未实测。

### 2.4 什么样的监督可以入库

| 检查项 | 验收办法 |
|---|---|
| 计划与格式 | 唯一 final、有效 JSON、与指定计划一致、步数一致；逐步合法且精确到达 |
| 中间状态 | 使用附录的固定状态行，提取路径／占据格列表，与独立规则回放逐步比对 |
| 理由与最短声明 | 对照边、覆盖格、不可逆移除和显式下界；正确终局不能抵销错误理由 |
| 自然语言 | 小批全审；全量检查可提取事实并抽查理由，记录审阅覆盖，不能宣称自由文本逻辑全部自动认证 |

原始输出、修订及失败原因均保存。修订仍不过时，先决定规则模板兜底或升级解释教师，再做全量；不能静默丢题或把不可靠解释当监督。每条样本至少保留题目来源／split、计划与最短认证、回放状态／证据、教师版本与生成设置、监督文本及验收结果。大数据与生成日志放 Git 外。

## 3. 怎样训练

### 3.1 固定起点

| 项目 | 设置 |
|---|---|
| 基座与更新 | 两任务各自从原始 Qwen2.5-1.5B-Instruct 全参数训练，不使用 LoRA |
| 学习率／有效 batch | 2e-5／32 条完整样本 |
| 训练预算 | 3 epoch，每 epoch 打乱并遍历所有扩展样本 |
| 调度 | 前 5% 更新 warmup，之后 cosine 降至峰值的 1/10 |
| 监督范围 | 解释、最终计划和 assistant 结束 token；输入／padding 标签为 -100 |

工程预设建议为 AdamW（weight decay 0.01、beta 0.9／0.999、epsilon 1e-8）、梯度裁剪 1.0；设备支持时用 BF16 与梯度检查点，不跨题 packing。首轮一个训练种子；设备、单卡 batch 和累积步数由显存试样确定并写入合同。

### 3.2 多解怎样计权

目标是**先对回答 token 平均，再对同题不同解平均，最后对物理题平均**；解更多、解释更长不提高该题权重。

实现：共 N 道题、M 条扩展样本，题 i 有 n_i 条监督；每条先算回答 token 平均交叉熵，再乘固定权重 `M/(N*n_i)`，对 batch 平均。不要按每个 batch 的权重和再次归一化；梯度累积／分布式与末 batch 也要保持这一目标。零步题按同样的物理题权重进入训练。

### 3.3 长度由真实文本确定

1. 用实际 tokenizer 和正式 chat template，统计**学生输入 P、完整回答 R、整条训练序列 T**，保存分位数、最大值和最长原文；按最终拼接计数。
2. 全量验收文本的最大 T 决定训练上限 **S**；不按 P99 截掉长题。实际 R 提供新生成上限 **G** 的起点，再用小训练的 validation 检查学生重复／触顶，正式评测前固定。
3. 每任务分别定 S／G；核对最长输入＋G 和 S 都在原生上下文与设备能力内。超限先修改表达粒度并重验，不能静默截断。未微调与 SFT 同 G，不用 test 答案调预算。

P 含 assistant 起始标记，R 含结束 token；S 限完整训练序列，G 限新生成 token。附录的短例子不能决定正式长度，教师长度也不是学生长度上界。

### 3.4 实现边界与资源

旧接口代码会冻结 base_model、保存 `adapter.pt`，并使用逐轮地图输入与 constant warmup；**不能只改参数范围就复用**。需要普通 causal LM 的整题入口、上述加权损失和完整模型保存／重载；复用环境规则、基础日志与 token 对齐思路。

在不复制题面的前提下，寻路约 4,192–16,192 条、积木约 9,000–33,000 条训练样本；普通题最多 48,000 次首轮教师生成，修订另计。全参数的优化器状态和长序列开销需用最长样本实测，不能按冻结接口估算。

## 4. 怎样评测与选模

每题一次贪心输出，无工具、无中间反馈、无重采。只解析唯一 final 中的 JSON，不从解释修补动作，不跳过非法步骤。

| 结果 | 统计口径 |
|---|---|
| 主指标：普通合法完整到达率 | 所有普通题为分母；全部动作合法并精确到达才成功，解析／格式错误和截断均计失败 |
| 效率与预算 | 合法且最短率、成功步数；另报 32 动作内成功率，便于对应旧系统预算 |
| 目标判断 | 零步空计划正确率单列，不混入普通成功率 |
| 诊断与成本 | 首个非法动作、逐步状态错误／缺失、解释抽查、生成 token、触顶与延迟；按距离、棋盘来源、目标非空和曝光分组 |

每 epoch 末评完整固定 validation，以普通到达率选 checkpoint；平分看合法最短率，再选较早权重。test 只评未微调基座与选定 checkpoint，不用于选教师、改格式或调长度。最终计划成功而理由错误时，任务记成功、解释诊断记错，分别报告。

## 5. 按什么顺序落地

| 阶段 | 输入与工作 | 产物及完成条件 |
|---|---|---|
| 1. 固定源题 | 当前保存清单与源合同 | 确定实际路径、ID／split／数量及零步评测来源；区分历史预览与正式数据 |
| 2. 验证监督流程 | 每任务 32 题；生成多解、回放证据、原模型解释 | 所有计划／状态核验，小批文本全审；记录解数、unknown、错误、最长文本和成本，定求解预算及失败处理 |
| 3. 制作全量语料 | 固定源题与通过试样的流程 | 每题有认证最短解，全部入库监督通过检查；未解决题、重复与曝光冲突逐项记账并处理 |
| 4. 验证训练入口 | 验收语料、最长样本和少量训练题 | 得到 token 统计／S、显存吞吐与累积设置；核对 mask、损失、参数确实更新及完整模型重载，少量过拟合能复现监督 |
| 5. 固定正式合同 | validation 生成试点 | 固定 G、设备／种子／输出目录及选模规则；提交两任务 configs，试样权重不计正式结果 |
| 6. 正式训练评测 | 原始基座、固定合同和语料 | 各训练 3 epoch，保存 validation 选模依据、两组 test 原始输出与逐题回放 |

阶段 1–5 补齐实际路径、解释可用性、长度和资源，之后才具备正式开跑条件。若原模型不能写出可靠解释或最短认证无法完成，先解决该失败，不靠删难题维持流程。实施时在 `experiments/llm_task_sft/` 建立 README、configs、src 和针对性 tests；正式参数以配置为权威。本轮只修订文档，尚未实施新生成或训练。

## 附录 A：完整 CoT 样例

下面是人工说明样例，采用真实规则，展示格式与推理内容。它们不是生产教师输出，也不能代表正式文本长度。

### A.1 寻路

**学生输入：**

```text
Find a valid path from node 0 to node 5 in this undirected graph.
Reaching the goal is the first priority; prefer fewer moves.
Use only listed edges. Visit each node at most once.
In <reasoning>, explain the constraints, plan, and checks.
In <final>, return JSON with path and steps.

Neighbors:
0: [1, 2, 3]
1: [0, 2, 4]
2: [0, 1, 5]
3: [0, 4, 5]
4: [1, 3, 5]
5: [2, 3, 4]
```

教师额外收到计划 `[0,2,5]` 及真实边／状态，学生不收到。

**监督输出：**

```text
<reasoning>
Goal constraints:
Connect 0 to 5 using listed edges without repeating a node.
There is no edge 0 -> 5, so at least 2 moves are needed.

Plan construction:
For a 2-move route, the middle node must neighbor both 0 and 5.
Their neighbor sets intersect at {2, 3}; either gives a 2-move route.
Use node 2. This does not mean node 3 is worse.

State checks:
Start at 0 with visited path [0].
Take 0 -> 2: the edge exists and 2 is unvisited.
Now at 2 with visited path [0, 2].
Take 2 -> 5: the edge exists and 5 is unvisited.
Now at 5 with visited path [0, 2, 5].

Verification:
Both edges exist, no node repeats, and the path ends at 5.
It matches the 2-move lower bound, so it is shortest.
</reasoning>
<final>
{"path": [0, 2, 5], "steps": 2}
</final>
```

另一条路线 `[0,3,5]` 需生成相应状态；允许的次优路线 `[0,1,4,5]` 可以说三步合法，不能声称匹配两步下界。正式图为原 256 节点图，不保证一般长题都有这样短的构造证明。

### A.2 积木

**学生输入：**

```text
Find a valid removal sequence from the initial board to the goal board.
Reaching the goal is the first priority; prefer fewer removals.
The board is a 10x10 binary grid: 1 is occupied and 0 is empty.
remove(shape_id,row,col) uses the shape's zero-based top-left anchor.
Every occupied shape cell must overlap an occupied board cell and lie
inside the board. Only those cells are removed. Empty shape cells
impose no constraint. Use only listed orientations. Shapes may be reused.
There is no gravity or inventory limit. Reach the goal exactly.
Shapes (/ separates rows):
0=11/10; 1=10/11; 2=11/01; 3=01/11; 4=1/1; 5=11; 6=1/1/1; 7=111.

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

In <reasoning>, explain the constraints, plan, and checks.
List all occupied cells after each removal; unlisted cells are empty.
In <final>, return JSON with actions and steps.
```

教师额外收到两个水平二格移除及逐步状态；单格无法覆盖的理由来自形状规则。

**监督输出：**

```text
<reasoning>
Goal constraints:
The goal keeps (9,9). Preserve it because removals cannot be undone.
Remove exactly [(0,0), (0,1), (0,2), (0,3)] using disjoint shapes.

Plan construction:
Every allowed shape removes two or three cells.
A three-cell removal here leaves one cell to remove.
No allowed shape fits inside a one-cell remainder, so that choice fails.
Two horizontal two-cell shapes cover the four cells without touching (9,9).
Use remove(5,0,0) and remove(5,0,2).
These placements are disjoint; their order can be exchanged under these rules.

State checks:
Initially occupied: [(0,0), (0,1), (0,2), (0,3), (9,9)].
remove(5,0,0) covers [(0,0), (0,1)], both occupied and outside the goal.
Occupied after removal: [(0,2), (0,3), (9,9)].
remove(5,0,2) covers [(0,2), (0,3)], both still occupied and outside the goal.
Occupied after removal: [(9,9)].

Verification:
Both removals are legal, preserve (9,9), and leave exactly the goal.
Four cells must be removed and one action removes at most three,
so at least two actions are needed. This two-action plan is shortest.
</reasoning>
<final>
{"actions": [{"shape_id": 5, "row": 0, "col": 0}, {"shape_id": 5, "row": 0, "col": 2}], "steps": 2}
</final>
```

排除三格方案是规则推导，不声称先真实探索再回溯。一般题中，每格都有覆盖候选不保证整体可解；奇数格也不证明无解，因为允许二格和三格形状。

## 附录 B：参考依据

- [STaR，§3](https://arxiv.org/html/2203.14465)：给答案提示补写理由，训练时移除提示，支持本方案的答案条件自生成形式；答案正确不保证理由正确。
- [SCoTD，§2–3](https://arxiv.org/html/2306.14050)：每题多条理由监督；OPT-1.3B 使用 2e-5、batch 32，作为本方案保守起点。其教师与任务不同，不能直接搬用效果。
- [DeepSeek-R1，B.3.3／B.4.3](https://arxiv.org/html/2501.12948v2)：参考多采样筛选与 cosine 衰减至 1/10；原文 1.5B 蒸馏采用 1e-4、batch 64、约 80 万样本，本方案不照搬其规模和长度。
- [Searchformer，§3／5.1](https://arxiv.org/html/2402.14083v2)：真实程序搜索轨迹是另一种监督，成本随轨迹增长。本轮选择路线解释，若后续定位到找解能力不足，再讨论搜索监督。
