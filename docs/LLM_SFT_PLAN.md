# 1.5B 整题 CoT SFT 基线方案

本文件记录 2026-10-11 对话中确认的实验设计及待决事项。研究进度统一维护在[项目状态页](PROJECT_STATUS_AND_TODO.md)。当前阶段为方案设计，尚未启动数据生成、训练或远端运行；本文件不是正式运行合同。

## 1. 实验目的与比较边界

希望比较：面对寻路与积木任务，直接对小型 LLM 做任务监督微调，能否达到或超过“冻结 LLM＋认知地图＋可训练 cross-attention 接口”的任务表现。

### 1.1 比较对象

| 条件 | 训练对象 | 地图与交互 | 用途 |
|---|---|---|---|
| 已有地图接口方法 | cross-attention 接口；LLM 与地图冻结 | 有认知地图，逐步接收真实状态与合法候选 | 完整方案参照 |
| 新整题 CoT SFT | Qwen2.5-1.5B-Instruct 全参数 | 无认知地图及读取接口，一次生成完整解答 | 本次主要训练条件 |
| 未微调的同基座 | 不训练 | 与新 SFT 使用相同整题输入和生成预算 | 衡量任务 SFT 的直接收益 |

已有 cross-attention 条件也使用 SFT，只是训练接口、冻结 LLM。新方案还改变了交互方式及监督内容，因此首先是完整方案对照，不能将结果差异单独解释为地图收益或解冻 LLM 的收益。

### 1.2 本轮范围

- **任务：**寻路与积木都做，分别从同一原始基座训练两个模型。这样与已有分别训练的接口条件对应，首轮不引入任务混合比例。
- **输入输出：**整题输入，一次输出 CoT 与完整计划；生成期间不接收环境逐步反馈。
- **监督来源：**求解器生成，每题采样多种合法、可到达目标的路径；首轮不使用大模型生成 CoT。
- **特殊类别：**首轮保留可解题和初始即目标的零步题，暂不加入整题无解监督。
- **尚未选择：**CoT 采用状态—动作轨迹、原始搜索日志，还是新增推荐的带依据搜索过程，见第 4 节；新增推荐不视为已确认条件。

## 2. 数据范围与划分

### 2.1 复用已有物理任务

**选择：复用现有物理训练题与训练／验证／测试划分，为训练题重新生成求解器示范。** 不因新方案改用求解器就重新抽取一批更容易或更难的题。

理由是先控制任务分布变化，使新旧方法的比较更容易解释。已有训练题曾按地图贪心成功筛选，复用会保留这一选题偏差；报告中应披露，不能视为对任务空间的无偏采样。

- 寻路保留现有同图、留出目标的边界；图的邻接表属于题目信息，不是被去掉的认知地图。
- 积木保留同初始棋盘新任务和新初始棋盘两类留出组，分别报告结果。
- 旧版本存在不同评测范围，正式配置需指定所复用的精确清单，不能混用历史数量与当前评测题集。

### 2.2 防止多路径引入泄漏

1. 先固定物理任务划分，再为训练题生成多条解；同题不同路径不能分入训练与测试。
2. 检查新增训练路径及显式监督的搜索分支中的 `(状态, 目标)` 是否与留出题重合，避免把留出题作为训练中的已解子问题直接示范。
3. 若发现冲突，优先重新采样训练路径，并保留冲突及未解决记录；不静默改动固定评测集。

### 2.3 输入呈现与增强

- 寻路输入完整图、起点、终点及不重复访问规则；积木输入规则、初始棋盘和目标棋盘。
- 使用适合整题的简单呈现增强，例如打乱邻接表行序及邻居显示顺序；具体模板与增强次数待定。
- 不原样保留旧逐轮候选编号增强。整题计划使用稳定的节点标识或物理动作表达，不依赖未提供的未来候选表。
- 原始解、重排输入和训练样本分别计数；呈现增强不增加独立物理题数。

## 3. 求解器与多路径采样

### 3.1 求解器的职责

求解器根据真实规则枚举动作、计算状态转移并搜索解，用于制作训练数据。测试时 LLM 独立生成整题答案，裁判在生成结束后验证计划；求解器不参与模型的在线行动选择。

寻路的不同解对应不同节点序列；积木的不同解可以对应不同移除顺序或形状组合。每条解都必须逐步合法，最终精确满足目标。

### 3.2 已接受的起始采样方案

| 项目 | 起始选择 | 理由 |
|---|---|---|
| 每题解数量 | 最多 4 条不同解 | 引入多样性，同时控制首轮数据规模 |
| 长度范围 | 至少保留一条最短解，其余解最多比最短解多 2 步 | 保留短解监督，也允许最短解数量少的题获得不同解法 |
| 解不足 4 条 | 保留实际找到的解，不删题、不复制凑数 | 避免偏向容易产生多解的题 |
| 去重单位 | 真实节点／动作序列 | 不把换编号或文字改写当成新路径 |
| 训练权重 | 按物理题平衡 | 防止多解题因样本多而获得更大权重 |
| 寻路合法性 | 不重复访问节点 | 沿用现有任务规则，不能靠绕圈制造多样性 |

“最多 4 条”和“最短步数加 2”是已接受的首轮起点，不是文献证明的最优值。后续样本统计需要报告每题有效解数量与长度分布；若多样性不足，再讨论调整，不自行扩大范围。

### 3.3 搜索失败和认证

- 搜索超时或节点预算耗尽只能记为未求解，不能标成无解。
- 最短性需由精确算法或已有可靠距离认证支持，不能把首次找到的路径自动称为最短解。
- 未获得所需认证的训练题单独记录，不静默删除后声称完整复用了原题集。
- 搜索算法、随机化方式、资源上限，以及独立动作换序与不同形状分解的多样性统计仍待细化。

## 4. 监督设计

### 4.1 检查结论：要补的是判断依据，不是推理字数

旧稿的搜索示例主要记录 `Pop / Try / Backtrack`。这些记录能够监督算法执行，但没有充分展示：目标施加了什么约束，为什么检查某个候选，凭什么排除一个分支，以及何时足以停止搜索。给这些操作增加泛泛的“我需要仔细思考”不能补上缺失的依据。

**新增建议：将带可验证依据的搜索 CoT 作为候选 B2。它是本次自查后的推荐，尚未获用户选择，不替换已确认的实验条件。**

| 候选 | 输出内容 | 适合检验什么 | 主要限制 |
|---|---|---|---|
| A：状态—动作轨迹 | 成功计划上的动作与状态变化 | 显式状态推演能否帮助完成任务 | 不教如何发现计划 |
| B1：原始搜索日志 | 实际展开、入队、候选尝试与回溯 | 模型能否模仿具体搜索程序 | 机械记录多，文本可能很长 |
| B2：带依据的搜索 CoT（推荐讨论） | 目标约束 → 搜索策略 → 候选及证据 → 排除／保留 → 计划验证 | 模型能否学会使用约束和搜索证据组织解答 | 需实现证据记录与文本模板，效果仍须实验验证 |

B2 不必罗列每次容器操作，但必须保留影响结论的搜索步骤及其依据。因此应称为“带依据的搜索过程”，不能把压缩后的解释冒称全部底层执行日志。原始日志可以留作离线核对。

### 4.2 寻路示例：解释搜索次序和停止依据

下面是六节点、每节点三个邻居的无向图，用于替代旧稿中过于简单的单一死路例子。它仍是说明格式的小题，不能代表正式 256 节点图的难度或长度。三种答案均属于一次 assistant 输出。

**共同输入正文：**

```text
Find a valid path from node 0 to node 5 in this undirected graph.
Reaching the goal is the first priority; among valid paths, prefer fewer moves.
Use only listed edges. Visit each node at most once.

Neighbors:
0: [1, 2, 3]
1: [0, 2, 4]
2: [0, 1, 5]
3: [0, 4, 5]
4: [1, 3, 5]
5: [2, 3, 4]
```

**A：状态轨迹输出。** 格式指令为：“In <reasoning>, show chosen edges and updated visited paths. In <final>, return JSON with path and steps.”

```text
<reasoning>
Current node: 0. Visited path: [0].
Take edge 0 -> 2. The edge exists and node 2 has not been visited.
Current node: 2. Visited path: [0, 2].
Take edge 2 -> 5. The edge exists and node 5 has not been visited.
Current node: 5. Visited path: [0, 2, 5].
The goal has been reached after 2 moves.
</reasoning>
<final>
{"path": [0, 2, 5], "steps": 2}
</final>
```

**B1：原始搜索日志输出。** 格式指令为：“In <reasoning>, run BFS in listed neighbor order, record each discovery and expansion, and stop when the goal is discovered. Discover each node once. In <final>, return JSON with path and steps.”

```text
<reasoning>
Discover 0: depth=0, parent=none.
Expand 0: discover 1, 2, 3 at depth=1 with parent=0.
Expand 1: skip discovered nodes 0 and 2; discover 4 at depth=2 with parent=1.
Expand 2: skip discovered nodes 0 and 1; discover 5 at depth=2 with parent=2.
Goal discovered. Follow parents: 5 <- 2 <- 0.
</reasoning>
<final>
{"path": [0, 2, 5], "steps": 2}
</final>
```

这里改用记录节点、深度与父节点的 BFS。对于本例的无权最短路，这避免了旧稿“把所有简单路径放进队列”的大量重复；枚举多条含次优的简单路径需要另一个生成步骤，不能用这个单解 BFS 代替。

**B2：带依据的搜索 CoT。** 格式指令为：

```text
In <reasoning>, identify the goal constraints, explain the search strategy,
and show the evidence for discoveries, exclusions, and stopping.
For this example, use BFS in listed neighbor order.
Claim shortestness only when a lower bound or completed search justifies it.
In <final>, return JSON with path and steps.
```

监督输出：

```text
<reasoning>
Goal and lower bound:
I need a path from 0 to 5 without repeated nodes.
Node 5 is not a neighbor of 0, so a solution needs at least 2 moves.

Search strategy:
All edges cost one move. I will expand nodes by increasing distance from 0.
Nodes at the same depth are considered in the listed neighbor order;
this order does not mean that an earlier node is closer to the goal.

Search:
Expanding 0 discovers nodes 1, 2, and 3 at depth 1.
For node 1, neighbors 0 and 2 have already been reached in fewer moves.
Its new neighbor is 4, reached by [0, 1, 4] in 2 moves.
This does not reach 5. It rules out a 2-move solution through node 1,
but it does not prove that all longer paths through node 1 fail.

Next, expand node 2 at depth 1.
Neighbors 0 and 1 already have shorter recorded routes.
Neighbor 5 is the goal, giving [0, 2, 5] in 2 moves.

Verification:
Both edges exist and no node repeats.
The path ends at 5 and matches the 2-move lower bound.
It is therefore a shortest path. I can stop without expanding the
remaining frontier to look for an improvement; other equally short
paths may still exist.
</reasoning>
<final>
{"path": [0, 2, 5], "steps": 2}
</final>
```

**新增的逻辑是什么：**同层搜索次序只是次序，不虚构为“更优”；某一长度下未达标不等于分支无解；找到路径后，借助明确的下界证明可以停止。没有证据时不能输出“这是最短解”。例如同题的 `[0,1,4,5]` 合法但长一步，若作为允许的次优训练样本，只能声称可达。

### 4.3 积木示例：从目标约束推导排除依据

下面沿用项目真实的 10×10 棋盘、八种形状和零起始坐标。初始第一行四格、右下角一格，目标只保留右下角。输出中的坐标集合列出全部占据格，未列位置为空；这种状态表达本身仍是模板提案。

**共同输入正文：**

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
```

**A：状态轨迹输出。** 格式指令为：“In <reasoning>, list occupied cells and the result of each chosen removal. In <final>, return JSON with actions and steps.”

```text
<reasoning>
Occupied cells: [(0,0), (0,1), (0,2), (0,3), (9,9)].
Choose remove(5,0,0). It removes [(0,0), (0,1)].
Occupied cells: [(0,2), (0,3), (9,9)].
Choose remove(5,0,2). It removes [(0,2), (0,3)].
Occupied cells: [(9,9)].
The board equals the goal after 2 removals.
</reasoning>
<final>
{"actions": [{"shape_id": 5, "row": 0, "col": 0}, {"shape_id": 5, "row": 0, "col": 2}], "steps": 2}
</final>
```

**B1：原始搜索日志输出。** 此处改为覆盖搜索：选最少候选覆盖的待移除格（并列取坐标较小者），按形状面积从大到小尝试覆盖它的动作。该思路对应仓库已有精确覆盖求解器，不再人为按任意 shape_id 顺序遍历所有动作。

格式指令为：“In <reasoning>, search over non-overlapping shape placements covering the cells to remove. Log the pivot cell, candidates, trials, failures, and backtracking. In <final>, return JSON with actions and steps.”

```text
<reasoning>
Remaining cells to remove: [(0,0), (0,1), (0,2), (0,3)].
Pivot: (0,0). Covering candidates: [remove(7,0,0), remove(5,0,0)].
Try remove(7,0,0). Remaining: [(0,3)].
Cell (0,3) has no covering candidate. Backtrack.
Try remove(5,0,0). Remaining: [(0,2), (0,3)].
Pivot: (0,2). Covering candidates: [remove(5,0,2)].
Try remove(5,0,2). Remaining: [].
Return [remove(5,0,0), remove(5,0,2)].
</reasoning>
<final>
{"actions": [{"shape_id": 5, "row": 0, "col": 0}, {"shape_id": 5, "row": 0, "col": 2}], "steps": 2}
</final>
```

**B2：带依据的搜索 CoT。** 格式指令为：

```text
In <reasoning>, identify cells to preserve and remove. Explain the search
choice, show the evidence for rejected candidates, and verify the plan.
Use only deductions supported by the rules and explored states.
In <final>, return JSON with actions and steps for the successful plan.
```

监督输出：

```text
<reasoning>
Goal constraints:
The goal keeps (9,9), so I must never remove it: removals cannot be undone.
The cells to remove are R = [(0,0), (0,1), (0,2), (0,3)].
A solution must cover R exactly with non-overlapping allowed shapes.

Search choice:
Every solution must remove (0,0). Only remove(7,0,0) and remove(5,0,0)
can cover this cell without using an empty or preserved cell.
I can consider these placements first without losing a solution:
the placements of any successful plan are disjoint, so their removal
order can be exchanged under these rules.
I try the three-cell placement first as a candidate for using fewer
moves; removing more cells immediately does not guarantee success.

Reject a candidate:
remove(7,0,0) leaves R = [(0,3)].
Every allowed shape occupies at least two cells, and no placement fits
inside this one-cell remainder. This branch cannot reach the goal.
Undo this trial in the search, restoring R.

Keep a candidate and complete:
remove(5,0,0) leaves R = [(0,2), (0,3)].
remove(5,0,2) covers exactly these two cells.
R is now empty; the occupied board is [(9,9)], exactly the goal.

Verification:
The two placements are disjoint, legal, and preserve (9,9).
Four cells had to be removed and one shape removes at most three,
so at least 2 removals were necessary. This plan uses 2 and is shortest.
</reasoning>
<final>
{"actions": [{"shape_id": 5, "row": 0, "col": 0}, {"shape_id": 5, "row": 0, "col": 2}], "steps": 2}
</final>
```

**新增的逻辑是什么：**先从“只能移除”推出目标格不可破坏；说明为什么围绕一个待移除格搜索不会漏解；用具体的单格残余证明失败；最后验证覆盖和步数下界。一般情况不能把“每格都有可覆盖动作”当作一定可解，也不能把“剩余格数是奇数”当作无解，因为本任务同时有二格和三格形状。

以上为经过规则检查的人工说明样例，尚不是已实现的自动文本生成器。正式生成必须遵守下一节的证据要求；不能把这个两步例子的易验证性推广到所有长任务。

### 4.4 带依据的文本如何自动生成

建议采用“求解器事件＋可核对证据＋固定语言模板”，不增加大模型教师。正式代码尚未实现，下面说明每类句子需要什么数据。

| 要写出的判断 | 必须有的证据 | 不能偷换的结论 |
|---|---|---|
| 某动作合法且保留目标 | 动作覆盖的格子／图中的边及当前状态 | 合法不等于后续一定可达 |
| 该分支无法完成 | 无候选、不可覆盖格等局部证据，或完整子搜索的失败证明 | 搜索超时不等于无解 |
| 优先试某个候选 | 明确的启发规则或同级排序规则 | 编号靠前、面积较大不等于真正更优 |
| 找到最短解 | 可解释的下界与解长度相等，或完成了所需最短性搜索 | 离线裁判知道最优长度，不代表文本已经推导出它 |
| 回溯或复用已解子问题 | 所退回的状态、失败分支或此前已建立的子问题结论 | 不能直接引用未在文本中建立的隐藏 oracle 结论 |

具体处理原则：

1. 保存目标约束、关键状态变化、候选选择规则、分支证据和最终验证；反复打印相同队列、棋盘或固定套话不作为“更多思考”。
2. 所有理由来自真实规则检查和搜索事件。解释应在其证据出现后给出，不能先知道答案再编一个发现过程。
3. 深层分支若失败但没有简短局部证据，需要保留支撑失败结论的子搜索或证明引用；不能压成一句没有依据的“不可达”。若因此过长，它是格式可行性问题，不能靠编造理由解决。
4. 搜索失败分支只出现在推理里，`<final>` 只含成功计划；整段始终是一次输出，没有外部执行反馈。
5. B2 更容易阅读、事实更容易逐项审查，是设计判断；是否比 A 或 B1 提高模型表现尚无本项目证据。格式选择继续待讨论。

### 4.5 本次可行性自查与需要修正的地方

| 问题 | 修正与仍需验证的部分 |
|---|---|
| 旧 BFS 按全部简单路径扩展，容易放大文本 | 单最短路用节点级 BFS；多解生成另行处理。BFS 的全局 visited 不能直接照搬到枚举所有简单路径 |
| 固定搜索次序通常只给出一个答案，却要求每题多解 | 可通过随机化同级候选得到部分不同最短解；允许的次优解需要有界枚举。每条搜索 CoT 必须对应真实生成该答案的过程；不能换掉最终路径后沿用旧推理 |
| 积木多路径可能只是动作排列 | 此任务只移除，成功计划中的移除集合两两不重叠且可交换。分别统计不同覆盖方案和同方案不同顺序；不把后者自动解释为不同规划策略。最多 4 条解的约定保持，类型比例尚未确定 |
| 现有求解器并不直接输出自然语言理由 | 已有 `blocks.py::solver` 提供精确覆盖求解及见证路径；仍需为覆盖约束、失败依据与搜索分支增加事件记录。不能把“能求解”写成“CoT 生成已完成” |
| 原始长日志加解释可能更长 | B2 可减少重复记账，但不能保证短于 B1；必须测实际训练题文本，尤其是深层失败分支 |
| 更多推理 token 不保证更好监督 | 过多模板句可能稀释动作及关键判断的损失占比。先检查有效判断内容与输出长度，不预设越长越好 |

非空目标的积木需要先检查目标占据格是初始占据格的子集，再求解“初始减目标”的待移除集合。该转换依赖本项目无重力、无库存、只能移除、形状空格无约束的规则；不能推广到一般 BlocksWorld 或 Sokoban。

### 4.6 样本结构与损失

```text
输入：任务规则＋完整题目
输出：CoT＋最终完整路径／移除序列
```

- **监督范围：**CoT 和最终计划全部计算交叉熵；输入提示不计算损失。
- **输出内部权重：**所有监督 token 等权，首轮不额外提高最终计划权重，减少人为超参。
- **样本聚合：**每条样本先对监督 token 求平均，再按物理题平衡。不同题的总训练权重不因路径数量或推理长度增加而自然放大；具体采样或加权实现需与此目标一致。
- **答案多样性：**同题多解分别构成训练样本；评测按执行结果判分，不要求逐字匹配某一条示范。

若完整搜索过程很长，应单独观察最终计划部分的学习情况；只有出现明确问题后，再讨论调整其损失权重。

### 4.7 零步与无解样本

- **保留零步题：**初始状态已经等于目标时，监督空计划和简短完成判断，不强制长 CoT。
- **暂不加入整题无解题：**首轮聚焦生成可达计划，无解证明可作为后续独立扩展。
- **不迁移旧失败标签：**原积木失败上下文只说明“这段历史之后，无合法动作且未达标”。原题可能有解，只是轨迹走入死局，不能改写成原始整题无解。
- 普通任务与零步题的采样比例仍需在正式合同中明确，不能因普通题多路径扩展而无意改变类别权重。

## 5. 训练设置与依据

### 5.1 文献实际设置

| 参考 | 模型与数据 | 已核实参数 |
|---|---|---|
| DeepSeek-R1 蒸馏，修订版附录 B.4.3、表 6 | Qwen2.5-Math-1.5B；约 80 万条混合数据 | 初始学习率 1e-4，batch 64，2–3 epoch，余弦衰减至初始值的 1/10，上下文上限 32,768 |
| SCoTD，§3 | OPT-1.3B；任务专属 CoT | 学习率 2e-5，batch 32，每题采样 30 条教师推理 |

来源：[R1 修订版](https://arxiv.org/html/2501.12948v2)、[SCoTD](https://arxiv.org/html/2306.14050)。R1 与我们同为 1.5B，但基座与数据规模不同；SCoTD 的任务主要是常识问答。它们提供超参起点，不保证在本任务上最优。Searchformer 用于参考数据生成，其从头训练超参不直接搬到 Qwen 微调。

### 5.2 已接受的本项目起始设置

| 项目 | 选择 | 理由与边界 |
|---|---|---|
| 参数更新 | 全参数微调 | 直接检验任务微调能力，首轮不额外引入 LoRA 容量限制 |
| 初始／峰值学习率 | 2e-5 | 参考 SCoTD，作为较小任务数据上的保守起点；并非断言优于 R1 的 1e-4 |
| 有效 batch | 32 条完整样本 | 可用梯度累积实现，不等于单卡 microbatch 32 |
| 训练预算 | 首轮 3 epoch | 参考短程 SFT 预算，不继承接口训练的 20 epoch；是否延长依据验证集另行讨论 |
| 调度 | warmup 后余弦衰减至峰值的 1/10 | 衰减方式参考 R1 |
| warmup | 总更新步数的前 5% | 本项目采用的起始设置，不冒称 R1 或 SCoTD 的已公开参数 |
| 最大序列长度 | 用实际求解文本确定，步骤见 §5.3 | 每个任务在格式选定后固定一个数值，不凭短示例或参考论文直接填写 |

优化器细项、数值精度、梯度裁剪、单卡 batch、随机种子等尚未确定；在实施前写入正式合同，不由框架默认值暗中决定。超出长度预算的样本需要报告并讨论处理方法，不能静默过滤后改变难度分布。

### 5.3 如何从求解文本确定长度上限

**用户已明确：长度上限根据具体求解文本确定。** 当前六个小示例用于比较格式，不代表正式任务难度，不能据此直接填 8K、16K 或 32K。搜索文本的长度取决于实际展开的分支及证明内容，不能只按最终动作数估算。

**第一步：先拿到可检查的文本。**

1. CoT 选择前，先确定需要小批比较的候选格式，再用正式训练任务中覆盖不同步数和棋盘来源的题生成实际求解文本，展示典型样本和最长样本。不默认三个候选都做完整实验。
2. 检查状态表达是否完整、搜索日志是否冗长、最终计划是否正确，再确定采用的格式与记录粒度。
3. 格式选定后，按该格式统计完整训练语料；小批数据用于估计成本，不能代替最终全量长度检查。校准长度使用训练数据，不能根据测试答案调预算。

**第二步：用实际 tokenizer 计数。**

使用 Qwen2.5-1.5B-Instruct 的 tokenizer 与正式 chat template，保留角色标记、格式指令、最终计划及结束 token，分别计数：

| 记号 | 具体计算什么 | 对应限制 |
|---|---|---|
| P | 推理时的完整输入前缀，包含 assistant 生成起始标记 | 输入占用 |
| R | 完整监督回答，包含 CoT、最终计划及结束标记 | 回答长度 |
| T | 套用 chat template 后的整条训练样本 | 训练序列长度 |

P、R、T 按实际模板拼接后的 token 边界计数，不能用字符数或独立分词片段的简单相加替代。每任务、每格式报告中位数、P95、P99、最大值及最长样本原文；另记最终动作数和搜索展开数，解释长文本来自哪里。

**第三步：按观测值提出一个固定上限。**

撤回旧稿中“最大回答长度乘 1.25”的默认提案：没有实际文本与生成记录，25% 余量没有任务依据。训练容量与模型实际生成预算应分别确定：

- **训练上限 S：**以选定格式的全量训练样本最大 T 为依据，必要时按实现需要取整。完整保留所有已纳入样本，不按 P99 静默截掉尾部。
- **生成上限 G：**先以实际求解文本的回答长度提出候选值；再在后续获准的小型训练试点中，用验证题检查模型是否因反复探索或重复表达而触顶，决定所需余量，并在正式评测前固定。教师文本长度只是参照，不保证学生输出同样长。生成限制只计新 token，不能与 S 混为一个参数。
- **上下文约束：**核对最长合法输入前缀加 G 不超过模型与运行时支持的上下文，并验证 S 可训练。若超限或成本不可接受，回到记录粒度或格式选择讨论，不能无声截断答案、删难题或自动扩大上下文。
- **固定方式：**寻路和积木分别确定 S、G，未微调与 SFT 条件使用相同 G。固定后不随单个测试题的已知答案长度调整，实际触顶仍计入失败诊断。

定稿时在正式配置写入实测数值及对应语料路径，本节只保留确定方法。目前尚未生成正式求解语料，也未进行实际 tokenizer 长度测量，因此没有可声称已经验证的数值上限。

### 5.4 预算如何计数

- 一个 epoch 指遍历扩展后的多解训练数据一次；例如每题 4 条解，训练 3 epoch，相当于每题约出现 12 次。
- 同时报告物理题数、不同解数量、呈现增强数量、训练 token 和参数更新步数。
- 新旧方法样本长度、训练参数量和信息条件不同，不能只凭 epoch 相同声称预算匹配。

## 6. 评测与模型选择

### 6.1 主协议

1. 每题一次贪心生成完整回答，不在生成期间给环境反馈，也不失败重采。
2. 裁判执行最终计划，要求每一步合法且最终精确达标；不跳过非法动作后继续计为成功。
3. 依据验证集的合法完整到达率选择 checkpoint；测试集不用于选择学习率、轮数或权重。
4. 未微调的同基座与 SFT 模型使用相同整题提示、解码设置及生成预算。
5. 多次采样成功率留作后续可选项，首轮不混入额外测试时采样收益。

### 6.2 指标与分组

| 类型 | 指标或分组 | 用途 |
|---|---|---|
| 主指标 | 合法完整到达率 | 判断能否独立完成整题 |
| 解质量 | 合法且最短解率、成功解步数 | 区分可达与高效 |
| 失败诊断 | 非法动作、输出截断、计划解析失败 | 定位失败环节，不能排除后再报成功率 |
| CoT 诊断 | 中间状态正确率；B2 中可核对的约束、失败证据与最短性声明 | 区分任务成功与解释正确性；自动核对结构化事实，抽查自然语言，不能声称已验证全部推理 |
| 类别分组 | 普通任务与零步题分别报告 | 避免简单终止题掩盖行动表现 |
| 积木留出组 | 同初始棋盘新任务、新初始棋盘分别报告 | 区分任务留出与棋盘泛化 |

已有地图接口结果作为完整方案参照，明确其逐步真实状态、合法候选与地图信息条件。整题输出不会产生同样的 reference／rollout 协议，不直接沿用旧逐轮指标作为主结果。

### 6.3 尚需定稿的评测细节

精确题集、验证频率、平分时的 checkpoint 选择规则和训练种子数量待定。训练与生成长度上限按 §5.3 的实际求解文本统计确定，不能照搬旧“每轮”输出上限当作整题预算。

## 7. 剩余决策与实施顺序

| 顺序 | 需要确定的内容 | 为什么此时确定 |
|---|---|---|
| 1 | 对照 §4.2–4.3 的具体文本讨论 A／B1／B2，明确需要比较的格式及小批求解算法 | 先看清模型要学习输出什么 |
| 2 | 所选候选的小批实际求解文本及 token 统计 | 比较实际长度、证据完整性和生成成本，再选 CoT 格式与记录粒度 |
| 3 | 选定格式的完整语料、解多样性及最大 token 长度 | 按 §5.3 确定每任务固定的训练与生成上限 |
| 4 | 类别权重、训练细项和完整评测合同 | 将已确认设计转成可执行设置 |
| 5 | 实现、正确性检查及正式运行安排 | 实施前固定合同和独立产物目录 |

上述是后续顺序，不代表已授权启动生成或训练。当前用户确认的是将推荐设计写入方案，CoT 选择仍保持开放。

实现优先复用现有 SFT 基础设施、环境规则和裁判；整题样本、监督 mask 与评测输入需按新协议调整。正式机器可读配置形成后，以配置为参数权威来源，本文件保留设计理由和比较边界。

## 8. 其他方法依据

- [Searchformer / Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/html/2402.14083v2)，§3：用 A* 实际搜索记录与最终计划构造监督，在迷宫和 Sokoban 中比较只输出计划与搜索过程加计划。其模型从头训练，不是本项目 Qwen 微调效果的直接证据。
- [A Close Look At World Model Recovery in Supervised Fine-Tuned LLM Planners](https://arxiv.org/html/2606.03685v1)，§3.2、表 1：研究在动作之间插入中间状态的 State-CoT；在其 BlocksWorld 和 Logistics 条件下收益不稳定，不能预设显式输出状态一定改善规划。其 BlocksWorld 与本项目积木消除规则不同。
- [SCoTD](https://arxiv.org/html/2306.14050)与[DeepSeek-R1 蒸馏修订版](https://arxiv.org/html/2501.12948v2)：参考其小模型 CoT SFT 超参，见第 5 节；本轮监督来源仍为求解器，不采用它们的教师模型生成流程。
