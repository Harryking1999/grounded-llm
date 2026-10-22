# 按任务工作的统一入口

不同环境使用不同的规则外壳和状态编码；共同过程是：识别当前状态、目标与合法候选后继，读取后继到目标的地图距离关系，把选择对应到本轮动作。先分别训练、评测，再考虑多任务泛化与合并。

## 任务能力总表

指标均为整题成功率，**每行明确题集与实际题数**。同一行未测试的模型用 `-` 占位；不同题集不横向相减。7B SFT为无地图基线；两列1.5B接图当前尚无正式对齐结果。

| 任务／目标范围／测试集 | DeepSeek | Qwen3-32B | 5.6 | 6.1 | 7B SFT | 认知地图本身贪心 | 1.5B＋认知地图：单任务训 | 1.5B＋认知地图：统一训 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **积木：12块清空，固定1000题（正式）** | - | - | - | - | - | **88.9%（889/1000）** | - | - |
| 积木：原文字40题，12块清空 | 77.5%（31/40） | 2.5%（1/40） | 92.5%（37/40） | 95%（38/40） | - | - | - | - |
| 寻路：原文字40题 | 97.5%（39/40） | 65%（26/40） | 100%（40/40） | 100%（40/40） | - | - | - | - |
| 寻路：固定留出256题 | - | - | - | - | - | **98.05%（251/256）** | - | - |
| 八盘汉诺塔：原文字40题 | 42.5%（17/40） | 10%（4/40） | 97.5%（39/40） | 95%（38/40） | - | - | - | - |
| 八盘汉诺塔：全状态到指定柱整塔，19680题 | - | - | - | - | - | **99.89%（19659/19680）** | - | - |
| 二阶魔方：原文字40题，复原 | 12.5%（5/40） | 2.5%（1/40） | 65%（26/40） | 77.5%（31/40） | - | - | - | - |
| 二阶魔方：全状态复原，3674160题 | - | - | - | - | - | **77.47%（2846397/3674160）** | - | - |
| 8-Puzzle：原文字40题，复原 | 22.5%（9/40） | 15%（6/40） | 82.5%（33/40） | 90%（36/40） | - | - | - | - |
| 8-Puzzle：全状态复原，181439题 | - | - | - | - | - | **98.28%（178326/181439）** | - | - |
| 手电过桥：首轮六人、留出32题，未采用 | 100%（32/32），Flash64k | - | - | - | - | 96.88%（31/32）；最优23/32 | - | - |
| 推箱子：首轮小棋盘、未见布局475起点，未采用 | - | - | - | - | - | 19.37%（92/475） | - | - |
| 推箱子：Flash64k筛查40题，同题地图未采用 | 100%（40/40） | - | - | - | - | 27.5%（11/40） | - | - |
| 24点：首轮XoT留出137题，未采用 | - | - | - | - | - | 51.09%（70/137） | - | - |
| 24点：Flash64k筛查40题，同题地图未采用 | 100%（40/40） | - | - | - | - | 55%（22/40） | - | - |
| Rush Hour：Flash64k，8个公开无墙分量40个起点，最短2–51步 | 42.5%（17/40）；最优17/40 | - | - | - | - | - | - | - |
| Token Swapping：Flash64k，公开8种架构40题，最优3–17步 | 90%（36/40）；最优36/40 | - | - | - | - | - | - | - |
| 推箱子：指定完整目标，Boxoban hard 10×10四箱，已测原题359、362 | 0%（0/2），Flash64k；2题截断 | - | - | - | - | - | - | - |
| Rush Hour：指定完整目标，8个公开无墙6×6原始深起点，最短38–51步 | 0%（0/8），Flash64k；8题截断 | - | - | - | - | - | - | - |
| Token Swapping：指定非归位目标，固定10顶点树，最优12–15步，基线地图未采用 | 80%（4/5），Flash64k；最优4/5 | - | - | - | - | 60%（3/5）；最优0/5 | - | - |
| Token Swapping：固定10顶点树，完整目标留出关系1024题，基线未采用 | - | - | - | - | - | 30.47%（312/1024）；最优73/1024 | - | - |
| Token Swapping：未见起点端点1024题，地图未采用 | - | - | - | - | - | 27.73%（284/1024）；最优75/1024 | - | - |
| Token Swapping：两端均未见1024题，地图未采用 | - | - | - | - | - | 27.25%（279/1024）；最优57/1024 | - | - |
| 推箱子：hard指定完整目标，留出关系1024题，基线未采用 | - | - | - | - | - | 14.65%（150/1024）；最优145/1024 | - | - |
| 推箱子：hard指定完整目标，未见布局1024题，基线未采用 | - | - | - | - | - | 6.84%（70/1024）；最优69/1024 | - | - |
| 推箱子：可达／边界采样对照，留出关系1024题，未采用 | - | - | - | - | - | 35.16%（360/1024）；最优341/1024 | - | - |
| 推箱子：可达／边界采样对照，未见布局1024题，未采用 | - | - | - | - | - | 6.64%（68/1024）；最优67/1024 | - | - |
| Rush Hour：指定完整目标，已见布局留出关系1024题，基线未采用 | - | - | - | - | - | 96.97%（993/1024）；最优829/1024 | - | - |
| Rush Hour：指定完整目标，未见布局1024题，基线未采用 | - | - | - | - | - | 50.78%（520/1024）；最优318/1024 | - | - |
| Token Swapping：补近阶段，留出关系1024题，未采用 | - | - | - | - | - | 48.54%（497/1024）；最优150/1024 | - | - |
| Token Swapping：补近阶段，未见起点1024题，未采用 | - | - | - | - | - | 47.85%（490/1024）；最优123/1024 | - | - |
| Token Swapping：补近阶段，两端均未见1024题，未采用 | - | - | - | - | - | 44.63%（457/1024）；最优127/1024 | - | - |
| Token Swapping：补近rank4，留出关系1024题，未采用 | - | - | - | - | - | 65.92%（675/1024）；最优181/1024 | - | - |
| Token Swapping：补近rank4，未见起点1024题，未采用 | - | - | - | - | - | 62.50%（640/1024）；最优175/1024 | - | - |
| Token Swapping：补近rank4，两端均未见1024题，未采用 | - | - | - | - | - | 64.36%（659/1024）；最优167/1024 | - | - |

过桥、推箱子、24点的地图列为[首轮固定目标可行性结果](../experiments/new_tasks_qmap/results/report.md)，不是正式采用图；Rush Hour与Token Swapping的完整状态对数据、训练与冻结reader已实现，尚无通过验收的地图。Flash筛查规则与结果见[基线报告](../experiments/task_text_baselines/results/report.md)；完整计划与逐轮环境反馈条件不同，不据差值直接归因。24点起题留出但部分中间状态与训练重叠；推箱子地图及固定40题使用自生成小棋盘，不是标准Boxoban题集。Rush Hour的40起点相关于8个分量，Token Swapping成本／最优性与完成率分别计分。

指定目标Flash三行来自[15题定向诊断](../experiments/task_text_baselines/results/exact_goals_flash64k.json)，沿用已知起点或等价重标问题，不是独立总体成绩。11份失败全为64k思考耗尽、最终答案为空；新条件下没有可回放的非法动作或非最优成功答案。后续只按精确目标训练与评测，旧目标成绩仅保留历史。Token Swapping地图数字为[完整预算冻结基线](../experiments/new_tasks_qmap/results/token_pairs_baseline.json)，尚未达到80%最低验收线；环境提供后继，不能与无反馈完整计划直接相减归因。

新增完整目标地图的完整预算结果：[推箱子](../experiments/new_tasks_qmap/results/sokoban_pairs_baseline.json)、[Rush Hour](../experiments/new_tasks_qmap/results/rush_hour_pairs_baseline.json)、[Token补近阶段](../experiments/new_tasks_qmap/results/token_pairs_local_goals.json)、[Token rank4](../experiments/new_tasks_qmap/results/token_pairs_local_rank4.json)。表内新增完整预算规划均为v2独特有序(S,G)查询，修正依据及原v1成绩见[去重核验](../experiments/new_tasks_qmap/results/planning_query_duplicate_audit.json)和结果的legacy字段；训练、验证选权重不变。各任务必需组尚未全部通过80%最低线，权重不登记采用；补近阶段与原基线分行保留，不能把增加训练与改变采样解释为单一因果。完整距离、循环、死锁、成本与资源见[报告](../experiments/new_tasks_qmap/results/report.md)。

**积木历史回归（单独题集，非12块清空1000题）：**

| 测试集 | 当前地图贪心4f413b5 | 旧地图贪心986cb4d | 1.5B＋旧地图：单任务SFT | 7B无地图SFT |
|---|---:|---:|---:|---:|
| 原地图SFT固定510题 | 84.90%（433/510） | 77.84%（397/510） | 94.12%（480/510） | 92.16%（470/510） |

旧1.5B为50k B2，7B为10k R＋A最佳断点，训练条件不同；不能据分差单独归因地图。旧1.5B到达且正确停止为475/510。统一训指同一个1.5B读图模型跨任务训练，各任务Qmap仍独立。

积木正式范围仅清空，**正式完成须≥900/1000，当前未完成**；[固定1000题](blocks/data/clear1000.json)、[验收合同](../experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json)。当前889/1000；相对880参照丢48、新增57，净增9，历史510为433（丢20、新增24）。789基线及全部历史权重保留。当前图原文字40题尚未补测；旧ffba71f在该40题为27/40，不填作当前图成绩。尚无1.5B／7B在1000题上的正式结果；独立最终泛化仍待验收，见[模型比较](../experiments/blocks_distance_map/results/blocks12_joint_report.md)。

采用权重：[积木](blocks/Qmap/selection.json)、[寻路](path/Qmap/selection.json)、[汉诺塔](hanoi/Qmap/selection.json)、[二阶魔方](cube2/Qmap/selection.json)、[华容道](puzzle8/Qmap/selection.json)。全状态评测包含地图训练状态，不称独立新状态泛化；原文字每任务40题、思考＋答案16k token，其他任务验收范围保持原定义。

证据：[当前积木1000逐题](../runs/qmap_dual_20261018/blocks12/decision_clear_028_300/evaluation/step_000300/results.json)、[纯文字四模型](../experiments/task_text_baselines/results/report.md)、[历史SFT](../experiments/map_sft_baselines/results/report.md)、[寻路](../experiments/path_qmap/results/summary.json)、[汉诺塔](../experiments/hanoi_qmap/results/full_coverage_summary.json)、[魔方](../runs/cube2_qmap/cycle_repair_f42c487/summary.json)、[华容道](../runs/qmap_dual_20261018/puzzle8/goal_expansion/stages/step_001000/metrics.json)。后续仅在相应题集行原位更新真实结果。

## 任务目录

| 任务 | 任务工作区 | 文字与提示 | 数据形态 | 地图 |
|---|---|---|---|---|
| 积木 | [blocks](blocks/README.md) | [prompts](blocks/prompts/README.md) | [data](blocks/data/README.md) | [Qmap](blocks/Qmap/README.md) |
| 寻路 | [path](path/README.md) | [prompts](path/prompts/README.md) | [data](path/data/README.md) | [Qmap](path/Qmap/README.md) |
| 汉诺塔 | [hanoi](hanoi/README.md) | [prompts](hanoi/prompts/README.md) | [data](hanoi/data/README.md) | [Qmap](hanoi/Qmap/README.md) |
| 二阶魔方 | [cube2](cube2/README.md) | [prompts](cube2/prompts/README.md) | [data](cube2/data/README.md) | [Qmap](cube2/Qmap/README.md) |
| 8-Puzzle（数字华容道） | [puzzle8](puzzle8/README.md) | [prompts](puzzle8/prompts/README.md) | [data](puzzle8/data/README.md) | [Qmap](puzzle8/Qmap/README.md) |
| 手电过桥 | [bridge](bridge/README.md) | [prompts](bridge/prompts/README.md) | [data](bridge/data/README.md) | [Qmap](bridge/Qmap/README.md) |
| 推箱子 | [sokoban](sokoban/README.md) | [prompts](sokoban/prompts/README.md) | [data](sokoban/data/README.md) | [Qmap](sokoban/Qmap/README.md) |
| 24点 | [game24](game24/README.md) | [prompts](game24/prompts/README.md) | [data](game24/data/README.md) | [Qmap](game24/Qmap/README.md) |
| Rush Hour | [rush_hour](rush_hour/README.md) | [prompts](rush_hour/prompts/README.md) | [data](rush_hour/data/README.md) | [Qmap](rush_hour/Qmap/README.md) |
| Token Swapping | [token_swapping](token_swapping/README.md) | [prompts](token_swapping/prompts/README.md) | [data](token_swapping/data/README.md) | [Qmap](token_swapping/Qmap/README.md) |

## 新任务候选：难度与地图关系

已保留的下一批地图范围为：推箱子Boxoban hard的10×10四箱、Rush Hour无墙6×6、Token Swapping固定10顶点树。前两项都由同一个任务Qmap读取完整棋盘并覆盖多布局，按未见布局检验；每题指定一个完整成功目标棋盘，学习到指定目标的状态对最小成本。Token Swapping先做固定树上的任意完整状态对距离，验收后再考虑12–15顶点。三项都保持当前＋单目标＋合法候选的读取接口；具体输入及实施边界见[统一Q方法](QMAP_METHOD.md#保留的新任务范围)和各任务Qmap，这些条件尚无采用权重。

复杂度结论针对规模可增长的任务族，不是固定6×6或10×10棋盘，也不保证随机小实例能难住Flash。当前需要同时有可复用题目、清晰动作、可核对真值和可训练规模。Rush Hour与Token Swapping已接入文字规则与裁判，[Flash64k筛查](../experiments/task_text_baselines/results/report.md)分别完成17/40、36/40，所有成功答案均最优；[合同](../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)将完成与最优成本分开。两者已具备Qmap训练与冻结评测入口，尚无采用图或本条件LLM训练，其余候选尚未采用。

| 候选／来源 | 难在什么目标 | 对本项目的价值与限制 |
|---|---|---|
| Rush Hour／[复杂度与规则建模论文](https://air.uniud.it/retrieve/22b19a17-19a9-4b13-a1e2-3b9126cce139/paper_29.pdf)、[公开代码](https://github.com/fogleman/rush) | 可变棋盘推广版的目标可达性为PSPACE完全；车只能沿自身轴线移动 | 优先考虑。动作可逆，同一车辆布局的可达分量内有对称最短路；适合固定出口目标，也可另设完整排列目标考察任意状态对。需要区分两种目标及单格／多格动作口径 |
| Token Swapping／[ESA2016](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2016.66)、[树上的困难性](https://arxiv.org/abs/2103.06707)、[公开实例及适配](token_swapping/data/README.md) | 一步交换图上相邻节点的棋子，最少交换次数NP-hard，即使底图是树 | 最适合直接检验任意起终点的距离。连通底图上所有排列互达；仅要求最终完成时可由通用流程求解，必须把成本／最优性列为目标。tket固定问题的上游序列不保证最短，本地另用精确搜索标定 |
| 冲突图渡河／[Csorba等，SIAM](https://doi.org/10.1137/080736661) | 狼羊菜推广成任意冲突图，求可完成运输的最小船容量NP-hard | 保留为过桥邻近候选。冲突关系改变可行状态与连通性，输入需有位置、船／管理者位置、冲突图与容量。不能把最小容量的复杂度直接套在固定容量、保证可解的子集；一般冲突图规则尚无本地实现 |
| 多岛多手电／[原题及本地核对](bridge/README.md#变体与适用性) | 维护人员与手电分布，人员有不同目的地；已核对的具体规则没有NP-hard结论 | 保留过桥语义、精确状态规模小，但现有来源为教学任务，不能标为已有论文benchmark |

Rush Hour现有题库由作者公开了距离、棋盘和可达分量大小。本项目独立完整BFS核对了两题：含一堵固定墙的60-move题所在分量只有2,332个状态、12辆车；无墙51-move题示例有4,780个状态、13辆车，最短距离与分量大小均和来源一致。[数据定义与条目](https://www.michaelfogleman.com/rush/)、[棋盘与见证](../experiments/task_text_baselines/results/new_tasks_harder_flash64k.json)。这里一次move是同一辆车沿固定方向滑动任意合法格数，成本均为1；改成逐格移动会改变距离，不能沿用原标签。文字筛查采用无墙原题及分量内的最短轨迹中间状态，见[实际数据口径](rush_hour/data/README.md)。输入编码需保留车辆身份、长度、方向、位置、墙与目标条件。出口任务终点是集合，不能把它的标签冒充完整排列之间的距离；完整状态对地图已按静态车辆配置分组准备，实际覆盖见[数据证据](../experiments/new_tasks_qmap/results/exact_pairs_data.json)。

现有推箱子本身在可变棋盘／箱数的推广下已为PSPACE完全，见[原始文献入口](https://webdocs.cs.ualberta.ca/~games/Sokoban/papers.html)；Boxoban四箱10×10只是其中有限的经验题集。它有不可逆推动与死路，而当前对称地图距离不能同时精确等于所有有向状态对距离。因而固定目标价值排序与任意状态对关系必须分别验收，不能因题目复杂就假定现有Qmap编码足够。

## 每个任务采用相同布局

```text
tasks/<task>/
  README.md             # 环境、已有代码、结果与正式配置的入口
  prompts/README.md     # 任务描述、规则、环境更新、各组输出与监督答案
  data/README.md        # 数据字段、准备代码与划分说明
  data/example.json     # 小型说明例；不是新增训练数据
  Qmap/README.md        # 共享方法、任务适配、训练与加载入口
  Qmap/selection.json   # 已采用权重的来源、适用范围和正式配置
  Qmap/checkpoints/     # Git 忽略：只放选定权重，不堆训练过程
```

根目录 [TASK_TEXTS.md](../TASK_TEXTS.md) 是所有任务文字的集中副本，由 `python -m tasks.export_texts` 从各任务的 `prompts/README.md` 与文字示例生成。修改任务内原文后重生成；不要直接编辑集中副本。`python -m tasks.export_texts --check` 检查是否同步。

## 文字分组

- 地图组／CoT 组：共享任务描述与规则，每轮收到真实环境更新。地图向量是地图组的独立输入，不把精确裁判距离放进文字。
- SOTA 组／32B 组：共享同一任务的描述与规则，一次输出完整剩余计划 JSON、无中途反馈。五任务文字已用于用户授权的纯文字基线，预算和题集见对应正式配置；不照搬旧整段计划结果充当匹配基线。
- 监督答案单列排序、Reason、Action。地图组是排序→R→A，无地图组删除排序行；当前为简短 Reason，长 CoT 尚未准备。

## 代码和权重怎么维护

`tasks/` 是日常阅读入口；公共实现放在 `grounded_llm/core/`，规则、数据适配与裁判放在 `grounded_llm/tasks/`，正式合同与运行编排放在 `experiments/`。任务独立编排 Qmap 训练与评测，SFT 使用公共训练基础，见 [Qmap 方法](QMAP_METHOD.md)和[源码结构](../README.md#代码依赖与接入)。新训练写入 Git 外的 `runs/`；确认采用的模型才放入该任务的 `Qmap/checkpoints/`，来源写入 `selection.json`。

项目进展与下一步只维护在 [状态页](../docs/PROJECT_STATUS_AND_TODO.md)。历史结果仍按原合同解释。
