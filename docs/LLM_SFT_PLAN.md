# 1.5B 整题 CoT SFT 基线方案

本文件记录已确认设计、实施建议和待实测项。当前进度统一维护在[项目状态页](PROJECT_STATUS_AND_TODO.md)。本轮完成方案修订；尚未启动新的教师生成、实现或训练。

## 1. 已确认方案与审阅结论

### 1.1 首轮流程

**求解器提供多条已验证路线 → 原始 Qwen2.5-1.5B-Instruct 分别生成带依据的自然解释 → 规则核对与文本审阅 → 对同一原始基座做整题全参数 SFT。**

寻路、积木都做，分别训练两个模型。学生一次输出 CoT 与完整计划，生成期间不接收环境反馈。

| 环节 | 输入 | 输出／责任 |
|---|---|---|
| 离线求解 | 固定训练题与真实规则 | 多条合法计划、最短认证 |
| 解释教师 | 题目＋一条计划＋该计划的状态证据 | 约束、方案依据、逐步推演与验证 |
| 监督验收 | 计划、证据、教师解释 | 核对动作／状态／理由；记录不通过内容 |
| 学生训练 | 只有题目 | 学习生成验收后的解释＋完整计划 |
| 整题评测 | 留出题，只有题目 | 模型自主生成；提交后由裁判回放 |

教师使用未做本项目 SFT 的原始 1.5B，先完成固定小批试样。它不必独立解出困难积木题，已有计划和证据降低了解释任务的难度。教师与学生初始权重相同，教师额外收到路线提示：新增信息来自求解器，而不是教师参数更大。因此宜称“答案条件下的自生成监督”，不宣称已经获得强教师能力。

### 1.2 深度审阅后的判断

- **可行性：**图与积木规则明确，计划可由程序认证；模型解释的可靠性和长度尚待实测。同模型生成解释有技术先例，不保证在本任务上有效。
- **可执行度：**当前方案明确了算法、样本字段、验收、损失和选模方法；实际源清单路径、长度、教师试样与训练资源尚未补齐，暂不能直接启动正式运行。
- **可读性：**主文只围绕一个已选择流程展开，保留两个完整示例；替代教师与完整搜索放在后续讨论，不再要求读者在三套格式间反复切换。

原稿的重要修正是：把“找到路线”“解释路线”“核对解释”分成三项工作。正确计划不能替代正确解释，带依据的解释也不能自动证明模型学会搜索。

### 1.3 实验回答什么

比较任务 SFT 能否改善 1.5B 的合法完整到达率，并与已有地图接口方案作系统级参照。

| 条件 | 训练对象 | 推理条件 | 用途 |
|---|---|---|---|
| 新整题 CoT SFT | 原始 1.5B 全参数 | 规则＋完整题目；单次输出 | 本轮主训练条件 |
| 未微调同基座 | 不训练 | 同一题面、格式和生成预算 | 衡量 SFT 的直接收益 |
| 已有地图接口 | 接口；LLM 与地图冻结 | 地图、逐步真实状态和合法候选 | 完整系统参照 |

新旧条件同时改变参数更新范围、交互、监督来源和信息条件，差异不能单独解释为地图收益。首轮不增加任务混合、在线工具或额外重采条件。“只监督最终计划”的消融可用于识别 CoT 收益，但不自动加入本轮。

## 2. 是否需要搜索

### 2.1 当前选择：离线求解保留，完整日志不强制

| 所谓搜索 | 首轮处理 | 原因 |
|---|---|---|
| 程序寻找正确计划 | 保留 | 保证积木覆盖、最短认证和多解 |
| 输出全部分支、队列、回溯 | 不要求 | 会额外训练搜索算法模仿，成本可能远大于计划 |
| 推理中比较候选、短前瞻、局部排除 | 有证据时保留 | 展示约束怎样影响方案，不需打印全部程序执行 |

一般难题可能仍需要内部搜索才能发现解。省略日志不等于取消规划难度。本轮监督主要教**目标约束、方案组织、状态推演和计划验证**，尚不足以单独证明学会了发现解或执行搜索。

### 2.2 已知路线后生成 CoT 的边界

教师输入为题目、已验证路线及证据；学生输入只有题目。路线提示、oracle 数值和逐步事实包不进入学生提示。

解释应能独立阅读：可以提出一个方案并逐步证明它可行，不写“按上面给定的答案”。不能冒称它是在没有路线提示时真实发现答案的过程。

- 每条路线生成自己的解释，动作、状态和终局必须对应。
- 不强迫每一步都有排他理由；多种方案都可行时，不能写其他候选都失败。
- 局部比较需有依据，例如三格移除留下单格，所有形状至少占两格，因此分支必败。
- 只有成功计划时，不编造先前探索、失败尝试或回溯。
- 最短值可放在离线认证元数据中；解释若声称最短，还要给出可核对的下界或证明。

这里的“思考”指有内容的推导与检查，不以反思措辞或字数作为质量指标。

### 2.3 何时再考虑搜索监督

若学生已能正确推演给定计划，却在自主生成时经常选择进入死局的动作，再讨论加入真实失败分支与回溯。两类能力需要分开验证，不能仅凭最终失败就定位为缺少搜索。

[Searchformer](https://arxiv.org/html/2402.14083v2) §3 用 A* 搜索记录加计划训练模型，属于程序模仿；§5.1 指出长搜索轨迹的训练成本可能很高。它支持保留这条后续路线，不要求本轮复制全部搜索监督。

## 3. 物理题与多路线生成

### 3.1 固定源题，不重新选题

读取当前地图接口实验保存的物理题清单，保留 task ID、状态、目标、split 和组别；重新制作整题标签。原训练题曾按地图贪心成功筛选，复用保留该偏差，报告时披露。

| 任务 | 来源边界 | 应保留内容 |
|---|---|---|
| 寻路 | `path_single_long.json` 对应的单图整轨迹清单 | 4,000 普通训练题＋192 零步题；目标划分 192/32/32 |
| 积木 | 当前 `blocks_kv_restart.json` 对应的保存清单 | 8,000 普通训练题＋1,000 零步题 |
| 积木普通测试 | `blocks_kv_evaluation.json` 指定的当前清单 | 510 题，分别报告两种棋盘来源 |

数量是源合同预期，必须核对实际清单。当前本地 `runs/path10_manifest.json` 只有 320 条旧记录，`blocks_long_manifest_preview.json` 是历史预览，均不能误作当前正式来源。确切源路径仍待实施时补齐。

首轮不加入 1,000 条固定失败历史：它们证明历史已走入死局，不证明原始整题无解。保留可解题和初始等于目标的零步题。

旧积木“已达标评测”中的 100 题属于 validation。若用于本方案选模，不能再称独立测试；零步测试需从保存清单确认独立题及曝光，不能把训练题改名为测试题。

### 3.2 已接受采样范围

- 普通题最多 4 条不同计划，至少一条最短，其余不超过最短＋2步。
- 不足 4 条时保留实际数量，不删题、不复制补足；零步题仅一条空计划。
- 按真实节点／物理动作序列去重，文字改写与重排输入不增加解数。
- 寻路不重复节点；积木逐步合法、精确到达目标。
- 物理题总训练权重相同，不因解更多或解释更长而放大。

### 3.3 生成算法建议

| 任务 | 最短路线 | 补充路线 | 关键限制 |
|---|---|---|---|
| 寻路 | 节点级 BFS 得距离与见证 | 在最短至最短＋2深度内枚举简单路径；同级次序随机化 | 路径内 visited；剩余图距离可作安全下界剪枝 |
| 积木 | 待移除集合上的最小精确覆盖 | 有界覆盖枚举，优先不同覆盖方案；不足时可用同方案不同动作顺序 | 分别记录覆盖多样性与动作换序 |

积木先检查目标是初始占据格的子集，再求解“初始减目标”的集合。当前规则只有移除、无重力和库存限制、形状空格无约束，因此成功放置互不重叠，顺序可交换。选一个待移除格，遍历可覆盖它的放置，再递归残余，可得到不同覆盖方案。动作换序不自动代表不同规划策略。

最短认证与补充路线采用不同预算；找到最多 4 条即可停止补充，不要求穷举。采样不是全部合法解上的均匀抽样。具体时间／展开上限根据小批实际成本确定。

预算耗尽记为 unknown。最短认证未完成需单列；补充搜索结束可以保留已取得的解。不能把超时标成无解，或静默删掉困难题。

### 3.4 泄漏与曝光

- 同题多解留在同一 split；训练提示不含留出答案。
- 检查计划中显式监督的非终止 `(状态,目标)` 与留出题重合；若有失败分支，其显式状态也需检查。
- 新增积木路线可能改变旧“未见状态”分组：测试物理题不变，但重新统计实际曝光。
- 共有终局／空棋盘本身不等于整题答案泄漏；零步训练与评测的完全同题重合要单列。
- 冲突先尝试重采训练路线；仍无法解决则记录，不静默更换评测集或只保留容易题。

## 4. 原模型解释教师与监督验收

### 4.1 教师设置和小批验证

**已选择原始 Qwen2.5-1.5B-Instruct，教师与学生从相同基座出发。** 不使用接口模型、已微调学生权重或测试题来生成训练解释。

建议先每任务 32 道固定训练题，覆盖不同距离、棋盘来源、非空目标及候选较多的题，审阅每题所选路线的文本；不是先筛容易题。原模型在部署可用设备上生成，暂不购买／调用另一教师。

小批记录：

- 各物理题／路线的解释验收率，不只统计最后一次成功输出。
- 状态错误、理由错误、路线改写、格式错误和冗长重复。
- 典型样本、最长原文、实际 token 与生成耗时。
- 从人工抽查判断它是否只重复动作，或能给出正确的约束／方案依据。

建议先生成一次，失败最多依据具体错误反馈修订一次；这是实施建议，需写入正式配置。失败不删物理题。若采用规则模板兜底，来源和占比必须显式记录；同模型解释是否足够、是否需要升级教师，由试样回答。

### 4.2 教师输入与格式指令

事实包来自规则回放，包括该计划的每步动作、边／覆盖格、后继状态、目标保持和终局；局部排除只提供已有证明的内容。它不是把全部合法后继或 oracle 距离交给学生。

```text
You are given a task, a verified candidate plan, and checked transition facts.
Write a self-contained explanatory solution for a student that sees only the task.

In <reasoning>:
- Identify the goal constraints.
- Explain useful structural deductions and how the proposed plan satisfies them.
- State each action's effect and resulting state using the supplied facts.
- Check the exact goal. Compare alternatives only with supporting evidence.
- Do not invent search attempts, backtracking, distance estimates, or failed branches.
- Claim shortestness only if you give an explicit valid proof here.

In <final>, return exactly the supplied plan in the required JSON format.
The candidate plan and fact records are teacher-only context.
```

学生只收到题目和输出格式要求，不能保留上述“已提供计划”指令。不要求获取不可见的内部推理；训练的是教师明确输出的解释文本。

### 4.3 验收规则

| 内容 | 如何核对 | 不得偷换 |
|---|---|---|
| 最终计划 | 与指定计划逐项一致，JSON、步数、合法性与终局正确 | 教师改成另一合法计划也要重新登记验收 |
| 中间状态 | 原计划独立回放，与规范化状态字段比对 | 最终正确不抵销中间错误 |
| 目标保持 | 各移除与目标格不相交，目标属于初始 | 依赖不可逆移除规则 |
| 必败候选 | 单格、无覆盖格等证据，或完整证明 | 超时、编号、低评分不证明必败 |
| 最短声明 | 文本中的下界匹配或足够的搜索证明 | 元数据知道最优值不等于文本已推导 |
| 自然语言理由 | 事实字段核对＋人工抽查 | 不能宣称全部自由文本逻辑已自动认证 |

小批解释全部审阅。全量优先约束到能对应结构化事实的表达，并抽查自然语言，分别报告计划、状态、已检查声明和语言审阅覆盖。

错误理由需修订或删除后重验。模型不能自己生成后只凭另一句“验证通过”验收。解释来源是答案条件生成，不作为模型真实发现路线过程的证据。

## 5. 两个完整监督示例

下面是人工说明样例，采用真实规则，展示格式与推理内容。它们不是生产教师输出，也不能代表正式文本长度。

### 5.1 寻路：局部构造，无需完整 BFS 日志

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

### 5.2 积木：有依据的排除不要求编造回溯

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

## 6. 样本合同、损失与长度

### 6.1 样本至少保存什么

```text
task_id, source_record_id, split, task_type, problem
plan_id, final_plan, shortest_steps, solution_steps, solution_family
plan_origin, intermediate_states, teacher_only_evidence
explanation_origin, teacher_model_revision, teacher_prompt, generation_settings
assistant_text, validation_results, language_review_status
```

不需要额外建哈希体系。原始计划、教师输出与生成日志放 Git 外；代码、紧凑配置与决策摘要进入仓库。student input 仅使用 problem 与固定格式指令。

零步题用空计划与简短相等判断，不强制长解释。首轮不加入整题无解；普通题与零步题单列统计。

### 6.2 损失与训练权重

已接受：输入和 padding mask 为 -100；解释、最终计划与 assistant 结束 token 全监督，回答内 token 等权。

目标是三级平均：

```text
对物理题平均 [
  对该题不同解平均 [
    对该回答监督 token 的交叉熵平均
  ]
]
```

建议每 epoch 遍历全部 M 条扩展样本一次并打乱。若有 N 个物理题、题 i 有 n_i 条样本，每条先算 token 平均损失，再乘固定权重 M/(N*n_i)，对 batch 平均。不要再按每个 batch 的权重和归一化。

这样零步总权重保持 192/4,192 和 1,000/9,000。不同解释长度与解数不放大题的权重；若改变类别比例，另列新参数。分布式／梯度累积需要实现同一目标，默认 batch token 平均不自动满足它。

首轮建议一条解配一份输入呈现，不复制旧六种候选编号。寻路行序／邻居序按保存种子打乱；评测呈现固定。文字增强不增加独立物理题或解数。

### 6.3 根据实际文本确定上限

1. 先取得 §4.1 小批真实解释，检查典型和最长样本，不凭本文件两步例子选择 8K／16K。
2. 用原始基座 tokenizer 与正式 chat template，统计输入 P、回答 R、整条训练序列 T；报告中位数、P95、P99、最大值与最长原文。
3. 完整语料生成后，按最大 T 确定训练上限 S，必要时按实现取整；不按 P99 静默删尾部。
4. 用实际 R 提出新生成上限 G，在获准的小训练试点中检查重复解释／触顶，再在正式评测前固定。教师长度不是学生长度上界。
5. 核对最长合法输入前缀＋G 不超运行时上下文，且 S 可训练。超限则回到表达粒度讨论，不默默截断、删难题或扩上下文。

P 包含 assistant 生成起始标记，R 包含完整解释、最终计划及结束 token，T 是套用 chat template 后的整条样本。按实际拼接后的 token 边界计数，不以字符数或独立片段 token 数简单相加代替；G 只限制新生成 token，S 限制完整训练序列。

每任务单独固定 S/G；未微调与 SFT 条件同 G。只用训练文本及验证试点校准，不根据测试答案调预算。目前没有正式语料或 tokenizer 长度实测值。

## 7. 训练与实现

### 7.1 已接受的起始超参

| 项目 | 设置 | 依据／边界 |
|---|---|---|
| 基座／更新 | Qwen2.5-1.5B-Instruct 全参数 | 两任务分别从原始权重训练 |
| 峰值学习率 | 2e-5 | 参考 SCoTD，任务不同 |
| 有效 batch | 32 条完整样本 | 单卡 batch×卡数×梯度累积 |
| 预算 | 3 epoch | 按扩展数据完整遍历；不自动延长 |
| 调度 | 前 5% 更新 warmup，之后 cosine 至峰值 1/10 | 衰减参考 R1；5% 为本项目起点 |
| S/G | 实际文本确定 | 见 §6.3 |

[DeepSeek-R1](https://arxiv.org/html/2501.12948v2) 附录 B.4.3、表6的 Qwen2.5-Math-1.5B 蒸馏为 1e-4、batch64、2–3 epoch、约80万混合样本、32,768上限；其基座与数据量不同。[SCoTD](https://arxiv.org/html/2306.14050) §3为 OPT-1.3B、2e-5、batch32、每题30条教师推理。本项目最多4条计划是已接受的起点，并非文献最优值。

工程细项建议，待写入正式合同：AdamW、weight decay0.01、beta=(0.9,0.999)、epsilon1e-8、clip norm1.0；支持时 BF16 和梯度检查点，不跨题 packing；保存数据与训练种子。首轮一个训练种子建立基线，不能报告训练方差。

### 7.2 旧接口代码不能只改参数

核对仓库代码：

- `train.py::build_reader` 构造地图接口，`fusion.py::MapReader` 会冻结 base_model。
- `MapSFTTrainer` 保存 `adapter.pt`；旧 collator／transcript 还包含地图与逐轮环境输入。
- 旧训练调度为 constant_with_warmup，不能依默认值实现本方案 cosine。

需要普通 causal LM 的整题训练入口，全部目标参数可训练，使用 §6.2 的损失，保存可独立加载的完整模型与 tokenizer。复用规则、基础日志和 token 对齐思路，不沿用旧地图标签与接口保存逻辑。

获准实施后在 `experiments/llm_task_sft/` 保存 README、configs、src、tests；当前不创建占位脚本。应具备的功能：

| 功能 | 产物 |
|---|---|
| 固定物理题导入 | 任务表、来源与split |
| 计划／证据生成 | 多解、回放、最短认证、未解决记录 |
| 原模型解释／验收 | 回答、修订与失败原因 |
| 编码与长度统计 | mask、token统计、S |
| 全参数训练 | 完整模型checkpoint |
| 整题评测 | 原始回答、逐题回放与汇总 |

### 7.3 资源与成本

不增加呈现复制时，训练样本量为寻路4,192–16,192、积木9,000–33,000，取决于实际解数。更新数据实际样本、3 epoch、有效batch32及末batch规则计算。

4,000＋8,000普通题、每题最多4条，对应最多48,000次首轮解释生成；修订另计，零步优先用规则短文本。使用原模型省去外部教师依赖，但本地推理仍消耗资源。

全参数训练的权重、梯度与Adam状态，以及长输入激活／logits，不能按冻结接口的开销估计。先用最长样本测前向／反向显存和吞吐，再定单卡batch、梯度累积、设备。远端实施前读取开发机手册并确认源产物位置。

## 8. 评测与选模

### 8.1 协议

- 每题一次贪心生成，无中间反馈，不重采。未微调和SFT用相同题面、格式、预算。
- 只解析唯一 `<final>` 内的JSON；不从解释中修补动作，不跳过非法步骤。
- 寻路核对起终点、全部边及无重复节点；积木逐步执行到精确目标。
- 重复／缺失final、步数不一致、解析失败和截断均保留为失败诊断。
- 建议每epoch末评完整固定validation，以普通题合法完整到达率选模；平分按合法最短率，再选较早checkpoint。零步单列。
- test只评未微调基座与选定SFT checkpoint，不用于挑教师、模板或长度。

动作预算必须与旧系统参照明确对应：建议同时报告32动作内成功与完整输出合法完成。允许的最短＋2如超旧预算，先说明冲突，不偷偷改变训练范围。

### 8.2 结果与诊断

| 指标 | 用途 |
|---|---|
| 普通合法完整到达率 | 全部普通题作分母，含截断、非法与解析失败 |
| 合法且最短率、成功步数 | 用精确裁判区分可达与效率 |
| 零步空计划正确率 | 单列目标判断 |
| 首个非法动作、状态错误 | 区分读题、转移推演与动作表达 |
| 解释正确性 | 已核对事实与自然语言抽查，不声称全逻辑自动认证 |
| token、触顶与延迟 | 解释计算成本与预算失败 |
| 距离、棋盘来源、目标非空、曝光 | 明确泛化范围 |

解释错误但最终计划成功，任务记成功、解释诊断记错；解释正确但计划失败，任务仍失败。寻路同图新目标不证明跨图能力，积木新初始棋盘不自动证明全部中间状态未见。

## 9. 替代教师与相关工作

### 9.1 为什么先用原模型

当前教师工作是解释已验证路线，而不是从零解决积木；收到证据后，原模型能否胜任可用很小试样判断。如果它只复述动作、频繁编造理由或修订仍无法通过，再讨论更强教师。

强教师可做两种不同工作：独立产计划，或只解释已有计划。前者按难题覆盖和成本评估，后者按理由正确率、有效内容和长度评估，不能以模型大小或语言流畅度代替检验。

若取消求解器产答案，可以用保存成功轨迹、强教师多采样或人工计划。但若连精确距离认证也取消，就不能保证“每题至少一条最短、其他≤最短＋2”，需重新讨论约定。

### 9.2 已有强教师证据与限制

[历史Flash对照](../experiments/gcml_counterexamples/README.md#deepseek-flash-comparison)及[摘要](../experiments/gcml_counterexamples/results/deepseek_flash.json)：八形状清空任务，8块115/128成功、13截断；12块74/128成功、54截断，两组含截断pass@8均16/16。已有能解积木的候选，不能断言没有强教师。

其所有尝试的输出token中位数约20,962／29,356，完成答案中位数20,120／23,427。成绩尚未验证当前非空目标题、自然解释质量及1.5B可学习性。历史Qwen3零分组用另一套十形状规则，不能直接否定当前教师。先审阅可复用文本，不开展大规模模型筛选。

### 9.3 文献采用的做法

| 工作 | 实际方法 | 本项目借鉴与边界 |
|---|---|---|
| [SCoTD](https://arxiv.org/html/2306.14050)，§2–3 | GPT-3每题多条理由＋答案，有标签时筛答案正确样本，训练小模型 | 多解释监督；主要是常识问答，未保证积木规划或全部理由正确 |
| [STaR](https://arxiv.org/html/2203.14465)，§3 | 自生成后筛正确答案；失败题给答案提示补理由，训练时移除提示 | 支持答案条件自生成；不能因答案正确就相信所有理由 |
| [Distilling Step-by-Step](https://arxiv.org/html/2305.02301)，§3–4 | 提取教师自然理由，标签预测与理由预测多任务训练 | 理由可作为监督；不同于本项目单次CoT＋计划，不直接搬收益 |
| [DeepSeek-R1](https://arxiv.org/html/2501.12948v2)，B.3.3、B.4.3 | 多采样、正确性与可读性筛选，再SFT／蒸馏 | 参考筛选与超参，不证明强教师规划正确 |
| [PDDL-Instruct](https://arxiv.org/html/2509.13351v1)，§5 | 状态—动作—状态生成，VAL核对前提／效果并反馈修订 | 外部验证与具体错误反馈；其多阶段协议不同 |
| [Searchformer](https://arxiv.org/html/2402.14083v2)，§3、5.1 | 程序搜索轨迹＋计划，学习搜索，再缩短轨迹 | 后续搜索监督参考，非本轮自然解释效果证据 |

原模型＋路线＋规则证据是本项目根据机制选择的组合；文献没有证明它在当前积木条件上优于强教师或规则模板。

## 10. 落地顺序与剩余事项

| 顺序 | 工作 | 可交付结果 |
|---|---|---|
| 1 | 确认真实源清单与评测ID | 固定物理题、split、曝光与分组 |
| 2 | 原模型小批路线解释 | 正确性、典型／最长文本、成本，明确修订与兜底政策 |
| 3 | 全量计划与验收后解释 | 覆盖、未解决题、解多样性、重复／泄漏记录 |
| 4 | 完整token统计与最长样本训练试样 | S/G候选、显存吞吐、设备与累积设置 |
| 5 | 小量题过拟合与保存重载检查 | 确认参数更新、mask与标签正确，验证生成触顶 |
| 6 | 固定并提交正式合同，执行3 epoch | 两任务模型、原始回答和固定评测结果 |

小量过拟合用于发现参数冻结、mask或拼接问题，不作为泛化成绩。检查只针对规则、目标保持、多解、损失权重、保存重载与裁判，不复制通用gate。

已确认主流程、原模型教师与训练起点；剩余项为实际源路径、修订／模板兜底、S/G、补充求解预算、工程细项、硬件与精确零步测试题。正式机器可读合同形成后，以配置为参数权威，本文件保留理由和边界。

本次授权范围是更新方案。新语料制作、实现、教师试样与GPU正式训练属于后续实施安排。
