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
- **尚未选择：**CoT 采用状态—动作轨迹还是完整搜索过程，见第 4 节。

## 2. 数据范围与划分

### 2.1 复用已有物理任务

**选择：复用现有物理训练题与训练／验证／测试划分，为训练题重新生成求解器示范。** 不因新方案改用求解器就重新抽取一批更容易或更难的题。

理由是先控制任务分布变化，使新旧方法的比较更容易解释。已有训练题曾按地图贪心成功筛选，复用会保留这一选题偏差；报告中应披露，不能视为对任务空间的无偏采样。

- 寻路保留现有同图、留出目标的边界；图的邻接表属于题目信息，不是被去掉的认知地图。
- 积木保留同初始棋盘新任务和新初始棋盘两类留出组，分别报告结果。
- 旧版本存在不同评测范围，正式配置需指定所复用的精确清单，不能混用历史数量与当前评测题集。

### 2.2 防止多路径引入泄漏

1. 先固定物理任务划分，再为训练题生成多条解；同题不同路径不能分入训练与测试。
2. 检查新增训练路径中的 `(状态, 目标)` 是否与留出题重合，避免把留出题作为训练轨迹的后半段直接示范。
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

### 4.1 CoT 内容：待选择讨论

**两种候选方案均未获选。不因状态轨迹较容易实现就自动采用，也不默认全部运行。**

| 候选 | 监督答案内容 | 能教给模型什么 | 代价与边界 |
|---|---|---|---|
| A：状态—动作轨迹 | 当前状态 → 动作 → 后继状态，最后汇总完整计划 | 沿一条成功解逐步维护状态、推演动作后果 | 容易从已验证路径生成；没有显式监督如何发现这条解、为何排除其他分支；中间状态会增加输出长度 |
| B：完整搜索过程 | 候选展开、分支探索、失败排除、回溯等实际搜索记录，最后输出完整计划 | 求解器如何找到解，包括未选中的探索分支 | 搜索记录可能远长于最终计划；需要定义记录粒度、搜索次序与长度处理，不能虚构搜索过程 |

待讨论的核心问题是：首轮主要让 LLM 学习成功解的状态推演，还是学习发现解的搜索程序。多路径采样与这一选择是两件事：同一道题有多条成功路径，并不意味着已经包含搜索过程监督。

无论选择哪种格式，推理时中间内容都由 LLM 自行生成。若记录状态、转移或分支判断，应由相应的真实环境或搜索记录支持；不能把仅核对终局视为已验证全部 CoT。

### 4.2 寻路：同一道题的两种具体输出

以下是用于讨论格式的完整小例子，不是正式训练集样本。正式寻路仍使用已有图和任务划分。示例采用英文，与已有实验语言一致；标签名称、标点及状态表示是待确认的模板提案，不表示 CoT 方案已选定。

**共同输入正文：**

```text
Find a valid path from node 0 to node 3 in this undirected graph.
Reaching the goal is the first priority; among valid paths, prefer fewer moves.
Use only the listed edges. Visit each node at most once.

Neighbors:
0: [1, 2]
1: [0]
2: [0, 3]
3: [2]
```

共同正文之后，根据条件追加 A 或 B 的格式指令。相同条件下，未微调模型与 SFT 模型收到完全相同的指令。

**A．状态—动作轨迹：只展示成功路径上的推演。**

输入追加：

```text
In <reasoning>, write the current node, each chosen edge, and the updated
visited path. In <final>, return JSON with path and steps.
```

监督输出：

```text
<reasoning>
Current node: 0. Visited path: [0].
Take edge 0 -> 2. This edge exists, and node 2 has not been visited.
Current node: 2. Visited path: [0, 2].
Take edge 2 -> 3. This edge exists, and node 3 has not been visited.
Current node: 3. Visited path: [0, 2, 3].
The current node equals the goal. Stop after 2 moves.
</reasoning>
<final>
{"path": [0, 2, 3], "steps": 2}
</final>
```

这个答案说明路径如何执行，没有说明为什么一开始不选节点 1。

**B．完整搜索过程：展示 BFS 找到同一条路径的过程。**

此例的具体算法是 BFS：按邻接表顺序扩展，用队列保存待探索路径，跳过当前路径中已出现的节点，在取出目标路径时停止。算法仅用于说明 B 的文本内容，正式求解器尚未选定。

输入追加：

```text
In <reasoning>, perform breadth-first search over paths. Show the queue,
each expansion, and skipped revisits. Stop when a goal path is popped.
In <final>, return JSON with path and steps.
```

监督输出：

```text
<reasoning>
Queue: [[0]].
Pop [0]. Node 0 is not the goal.
Append [0, 1] and [0, 2]. Queue: [[0, 1], [0, 2]].
Pop [0, 1]. Node 1 is not the goal.
Skip neighbor 0: it already occurs in this path.
No extension remains for [0, 1]. Queue: [[0, 2]].
Pop [0, 2]. Node 2 is not the goal.
Skip neighbor 0: it already occurs in this path.
Append [0, 2, 3]. Queue: [[0, 2, 3]].
Pop [0, 2, 3]. Node 3 is the goal.
Return [0, 2, 3], with 2 moves.
</reasoning>
<final>
{"path": [0, 2, 3], "steps": 2}
</final>
```

B 多监督了队列维护和无效分支处理。这里的“完整”指从初始化到找到该解的完整搜索记录，不要求找到解后继续穷举所有其他解。

### 4.3 积木：同一道题的两种具体输出

这个小例子遵守项目真实的 10×10 棋盘、八种形状和零起始坐标规则。初始棋盘第一行有四格，右下角有一格；目标仅保留右下角。下面的推理用 `Occupied cells` 列出所有占据坐标，未列坐标全为空。这是完整状态的稀疏表达，不是省略中间棋盘；是否采用这种表示仍属于模板讨论。

**共同输入正文：**

```text
Find a valid sequence of removals from the initial board to the goal board.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.
The board is a 10x10 binary grid: 1 means occupied and 0 means empty.
remove(shape_id,row,col) places the shape's top-left anchor at (row,col).
Coordinates are zero-based. Every occupied shape cell must overlap an occupied
board cell and lie inside the board. Remove only those cells.
Empty shape cells impose no constraint. Use only the listed orientations.
Shapes may be reused. There is no gravity or inventory limit.
Reach the goal board exactly, preserving its occupied cells.
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

**A．状态—动作轨迹：直接推演两次正确移除。**

输入追加：

```text
In <reasoning>, list all occupied cells, then each chosen removal and the
resulting occupied cells. Unlisted cells are empty.
In <final>, return JSON with actions and steps.
```

监督输出：

```text
<reasoning>
Occupied cells: [(0,0), (0,1), (0,2), (0,3), (9,9)].
Goal occupied cells: [(9,9)].
Choose remove(5,0,0). It removes [(0,0), (0,1)], both occupied.
It does not remove the goal cell.
Occupied cells: [(0,2), (0,3), (9,9)].
Choose remove(5,0,2). It removes [(0,2), (0,3)], both occupied.
It does not remove the goal cell.
Occupied cells: [(9,9)].
The board equals the goal. Stop after 2 removals.
</reasoning>
<final>
{"actions": [{"shape_id": 5, "row": 0, "col": 0}, {"shape_id": 5, "row": 0, "col": 2}], "steps": 2}
</final>
```

这条答案只包含成功解上的状态与动作，并不声称模型比较过其他移除方式。

**B．完整搜索过程：先遇到死局，再回溯找到解。**

为明确示例中的次序，此例采用深度优先搜索，候选按 shape_id 从大到小、再按 row 和 col 从小到大排列；跳过会移除目标格的动作，首次到达目标就停止。它与寻路示例采用不同算法，说明 B 可以有不同的求解器来源，不预设两个任务必须共用算法。

输入追加：

```text
In <reasoning>, perform depth-first search. Try legal removals in descending
shape_id, then ascending row and col order. Skip removals of goal cells.
List all occupied cells after each trial. Backtrack when no continuation
can reach the goal, and stop at the first solution.
In <final>, return JSON with actions and steps for the successful branch only.
```

监督输出：

```text
<reasoning>
Occupied cells: [(0,0), (0,1), (0,2), (0,3), (9,9)].
Goal occupied cells: [(9,9)].
Ordered legal removals: [remove(7,0,0), remove(7,0,1), remove(5,0,0), remove(5,0,1), remove(5,0,2)].
Try remove(7,0,0).
Occupied cells: [(0,3), (9,9)].
The goal has not been reached, and no legal removal remains. Backtrack.
Restore occupied cells: [(0,0), (0,1), (0,2), (0,3), (9,9)].
Try remove(7,0,1).
Occupied cells: [(0,0), (9,9)].
The goal has not been reached, and no legal removal remains. Backtrack.
Restore occupied cells: [(0,0), (0,1), (0,2), (0,3), (9,9)].
Try remove(5,0,0).
Occupied cells: [(0,2), (0,3), (9,9)].
Ordered legal removals: [remove(5,0,2)].
Try remove(5,0,2).
Occupied cells: [(9,9)].
The board equals the goal. Return the successful branch with 2 removals.
</reasoning>
<final>
{"actions": [{"shape_id": 5, "row": 0, "col": 0}, {"shape_id": 5, "row": 0, "col": 2}], "steps": 2}
</final>
```

该例的动作、状态、候选和死局可由仓库现有积木规则核对，成功计划最短为 2 步。深度优先“首次找到解”一般不保证最短，本例的最短性由独立精确求解确认。

**两种格式共同的边界：**整段文本都是一次 assistant 输出。B 中失败尝试和回溯是离线思考，不是真实环境执行，也不进入 `<final>` 计划。示例展示的顺序是具体算法顺序，不代表正式多路径采样策略已确定；正式 B 标签必须取自实际搜索日志，不能给现成成功路径编造探索经过。

### 4.4 样本结构与损失

```text
输入：任务规则＋完整题目
输出：CoT＋最终完整路径／移除序列
```

- **监督范围：**CoT 和最终计划全部计算交叉熵；输入提示不计算损失。
- **输出内部权重：**所有监督 token 等权，首轮不额外提高最终计划权重，减少人为超参。
- **样本聚合：**每条样本先对监督 token 求平均，再按物理题平衡。不同题的总训练权重不因路径数量或推理长度增加而自然放大；具体采样或加权实现需与此目标一致。
- **答案多样性：**同题多解分别构成训练样本；评测按执行结果判分，不要求逐字匹配某一条示范。

若完整搜索过程很长，应单独观察最终计划部分的学习情况；只有出现明确问题后，再讨论调整其损失权重。

### 4.5 零步与无解样本

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

**用户已明确：长度上限根据具体求解文本确定。** 当前四个小示例用于比较格式，不代表正式任务难度，不能据此直接填 8K、16K 或 32K。完整搜索文本的长度取决于实际展开的分支数量，也不能只按最终动作数估算。

**第一步：先拿到可检查的文本。**

1. CoT 选择前，用正式训练任务中覆盖不同步数和棋盘来源的小批题，分别生成 A、B 两种实际求解文本，展示典型样本和最长样本。此时不运行模型训练。
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

以下是便于定稿的计算提案，25% 余量和 256-token 取整属于工程建议，不是已有实测结论：

- **训练上限 S：**将全量训练样本的最大 T 向上取整到 256 的整数倍。完整保留所有已纳入样本，不按 P99 静默截掉尾部。
- **生成上限 G：**将最大 R 的 1.25 倍向上取整到 256 的整数倍，给模型生成留少量余量。生成限制只计新 token，不能与 S 混为一个参数。
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
| CoT 诊断 | 中间状态正确率（若输出状态） | 区分状态推演与最终计划正确性 |
| 类别分组 | 普通任务与零步题分别报告 | 避免简单终止题掩盖行动表现 |
| 积木留出组 | 同初始棋盘新任务、新初始棋盘分别报告 | 区分任务留出与棋盘泛化 |

已有地图接口结果作为完整方案参照，明确其逐步真实状态、合法候选与地图信息条件。整题输出不会产生同样的 reference／rollout 协议，不直接沿用旧逐轮指标作为主结果。

### 6.3 尚需定稿的评测细节

精确题集、验证频率、平分时的 checkpoint 选择规则和训练种子数量待定。训练与生成长度上限按 §5.3 的实际求解文本统计确定，不能照搬旧“每轮”输出上限当作整题预算。

## 7. 剩余决策与实施顺序

| 顺序 | 需要确定的内容 | 为什么此时确定 |
|---|---|---|
| 1 | 对照 §4.2–4.3 的具体文本讨论 A／B，明确小批求解算法及训练题来源 | 先看清模型要学习输出什么 |
| 2 | 两种格式的小批实际求解文本及 token 统计 | 比较实际长度和生成成本，再选 CoT 格式与记录粒度 |
| 3 | 选定格式的完整语料、解多样性及最大 token 长度 | 按 §5.3 确定每任务固定的训练与生成上限 |
| 4 | 类别权重、训练细项和完整评测合同 | 将已确认设计转成可执行设置 |
| 5 | 实现、正确性检查及正式运行安排 | 实施前固定合同和独立产物目录 |

上述是后续顺序，不代表已授权启动生成或训练。当前用户确认的是将推荐设计写入方案，CoT 选择仍保持开放。

实现优先复用现有 SFT 基础设施、环境规则和裁判；整题样本、监督 mask 与评测输入需按新协议调整。正式机器可读配置形成后，以配置为参数权威来源，本文件保留设计理由和比较边界。

## 8. 其他方法依据

- [Searchformer / Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping](https://arxiv.org/html/2402.14083v2)，§3：用 A* 实际搜索记录与最终计划构造监督，在迷宫和 Sokoban 中比较只输出计划与搜索过程加计划。其模型从头训练，不是本项目 Qwen 微调效果的直接证据。
- [A Close Look At World Model Recovery in Supervised Fine-Tuned LLM Planners](https://arxiv.org/html/2606.03685v1)，§3.2、表 1：研究在动作之间插入中间状态的 State-CoT；在其 BlocksWorld 和 Logistics 条件下收益不稳定，不能预设显式输出状态一定改善规划。其 BlocksWorld 与本项目积木消除规则不同。
- [SCoTD](https://arxiv.org/html/2306.14050)与[DeepSeek-R1 蒸馏修订版](https://arxiv.org/html/2501.12948v2)：参考其小模型 CoT SFT 超参，见第 5 节；本轮监督来源仍为求解器，不采用它们的教师模型生成流程。
