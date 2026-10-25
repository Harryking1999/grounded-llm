# 按任务工作的统一入口

本页索引已有十个任务，其中九类组成下一阶段任务池，汉诺塔保留为已有能力与后续扩展项。四任务联合SFT已经开展，实验范围与下一阶段选择见[项目状态](../docs/PROJECT_STATUS_AND_TODO.md)。各任务独立训练地图；多任务LLM共用读取接口。

## 多任务扩展准备情况

下一阶段新增推箱子、移车出库、棋子交换、过桥和24点。数据与训练设计统一见[Qmap方法](QMAP_METHOD.md#任务扩展与训练安排)；本表维护已有条件和接入缺口。

| 任务 | 已有地图与LLM条件 | 进入下一轮要补齐什么 |
|---|---|---|
| 积木、二阶魔方、8-Puzzle、寻路 | 各有采用地图，已进入四任务联合SFT | 补多目标结构验收；寻路准备新图及新Qmap；复用联合数据与评测入口 |
| 汉诺塔 | 八盘整塔目标地图已采用，本轮联合SFT未纳入 | 作为已有能力保留，后续按任务组合安排 |
| 手电过桥 | 五人、六人限定范围地图分别通过完成与成本验收并采用；见[报告](../experiments/new_tasks_qmap/results/bridge/report.md)及[selection](bridge/Qmap/selection.json) | 已采用两张地图的LLM读取与目标停止 |
| 24点 | 完整有理数DAG、多层目标、跨起题数值曝光与reader已有；四数原起点执行仍弱，见[报告](../experiments/new_tasks_qmap/results/game24/report.md) | 可达分支与原起点拟合；通过后再扩操作数规模 |
| 推箱子、移车出库、棋子交换 | 原树棋子交换已通过完成率验收并采用；推箱子、移车未通过 | 推箱子先验收固定墙体内多目标执行，移车保留跨布局；棋子交换分别验收固定树长程与新树地图，再接LLM |

Qmap、文字协议、正式SFT数据和LLM结果是不同准备阶段。现有公共接口可复用，下一轮任务组合由[状态页](../docs/PROJECT_STATUS_AND_TODO.md#推进安排)维护，各地图身份仍以selection为准。

## 任务能力总表

指标均为整题成功率，**每行明确题集与实际题数**。同一行未测试的模型用 `-` 占位。7B SFT为无地图基线。四任务固定500题的150K比较见[联合SFT报告](../experiments/map_sft_baselines/results/report.md#7-flan-150k重评测冻结地图的同题贪心参照)。

| 任务／目标范围／测试集 | DeepSeek | Qwen3-32B | 5.6 | 6.1 | 认知地图本身贪心 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **积木：12块清空，固定1000题（正式）** | - | - | - | - | **88.9%（889/1000）** |
| 积木：原文字40题，12块清空 | 77.5%（31/40） | 2.5%（1/40） | 92.5%（37/40） | 95%（38/40） | - |
| 寻路：原文字40题 | 97.5%（39/40） | 65%（26/40） | 100%（40/40） | 100%（40/40） | - |
| 寻路：固定留出256题 | - | - | - | - | **98.05%（251/256）** |
| 八盘汉诺塔：原文字40题 | 42.5%（17/40） | 10%（4/40） | 97.5%（39/40） | 95%（38/40） | - |
| 八盘汉诺塔：全状态到指定柱整塔，19680题 | - | - | - | - | **99.89%（19659/19680）** |
| 二阶魔方：原文字40题，复原 | 12.5%（5/40） | 2.5%（1/40） | 65%（26/40） | 77.5%（31/40） | - |
| 二阶魔方：全状态复原，3674160题 | - | - | - | - | **77.47%（2846397/3674160）** |
| 8-Puzzle：原文字40题，复原 | 22.5%（9/40） | 15%（6/40） | 82.5%（33/40） | 90%（36/40） | - |
| 8-Puzzle：全状态复原，181439题 | - | - | - | - | **98.28%（178326/181439）** |

三个新任务仍处于地图验证阶段，单列如下，完整结果见[三任务报告](../experiments/new_tasks_qmap/results/report.md)。

### 三个新任务

下表均为指定完整目标、冻结地图逐步贪心的结果。各主要组1,024题；每个任务使用同一行所指方案。

| 任务 | 熟悉环境的新关系 | 未见目标或状态 | 未见布局 | 状态 |
|---|---:|---:|---:|---|
| [推箱子](sokoban/README.md)，10×10四箱，1,000张训练棋盘 | 19.43%（199/1024） | 未见目标16.21%（166/1024） | 37.89%（388/1024），100张测试棋盘 | 未达标 |
| [移车出库](rush_hour/README.md)，无墙6×6 | 98.24%（1006/1024） | 未见目标98.83%（1012/1024） | 54.59%（559/1024） | 未达标 |
| [棋子交换](token_swapping/README.md)，原固定10顶点树L1 | 97.46%（998/1024），原保护关系回归 | 新查询未见起点98.24%（1006/1024）；两端未见97.46%（998/1024） | 新树独立地图亦通过 | 完成率通过；最优率另列 |

推箱子使用固定距离区间的同题重评：64张组与1,000张组的未见布局完成率为29.88%与37.89%，21步及以上均0/312；旧16张布局测试另保留，各组距离分布不同，覆盖量对照在同一题集上比较。

各必需组分别至少80%，目标90%以上，困难原题也需单独验收。棋子交换原树已采用，推箱子和移车未采用；三类均无LLM读取结果。指定目标的DeepSeek Flash 64k诊断为推箱子0/2、移车出库0/8、棋子交换4/5；这是已知困难题，见[文字结果](../experiments/task_text_baselines/results/exact_goals_flash64k.json)。

早期手电过桥、小棋盘推箱子、24点的地图结果分别为31/32、92/475、70/137；规则和题集不同，只保留为[首轮试验](../experiments/new_tasks_qmap/results/pilot/summary.json)。历史文字筛查见[文字报告](../experiments/task_text_baselines/results/report.md)。

**积木历史回归（单独题集，非12块清空1000题）：**

| 测试集 | 当前地图贪心4f413b5 | 旧地图贪心986cb4d | 1.5B＋旧地图：单任务SFT | 7B无地图SFT |
|---|---:|---:|---:|---:|
| 原地图SFT固定510题 | 84.90%（433/510） | 77.84%（397/510） | 94.12%（480/510） | 92.16%（470/510） |

旧1.5B为50k B2，7B为10k R＋A最佳断点，训练条件不同；旧1.5B到达且正确停止为475/510。统一训指同一个1.5B读图模型跨任务训练，各任务Qmap仍独立。

积木正式范围仅清空，**正式完成须≥900/1000，当前未完成**；[固定1000题](blocks/data/clear1000.json)、[验收合同](../experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json)。当前889/1000；相对880参照丢48、新增57，净增9，历史510为433（丢20、新增24）。789基线及全部历史权重保留。当前图原文字40题尚未补测；旧ffba71f在该40题为27/40，尚无1.5B／7B在1000题上的正式结果；独立最终泛化仍待验收，见[模型比较](../experiments/blocks_distance_map/results/blocks12_joint_report.md)。

采用权重：[积木](blocks/Qmap/selection.json)、[寻路](path/Qmap/selection.json)、[汉诺塔](hanoi/Qmap/selection.json)、[二阶魔方](cube2/Qmap/selection.json)、[华容道](puzzle8/Qmap/selection.json)。全状态评测包含地图训练状态，原文字每任务40题、思考＋答案16k token，其他任务验收范围保持原定义。

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

## 任务来源与结构背景

推箱子、移车出库和棋子交换已有完整目标地图实验；过桥和24点已有首轮实现与结果，尚无采用权重。过桥、24点按正常难度推进。下表保留任务来源与后续变体的结构背景；复杂度针对可增长任务族。

| 候选／来源 | 难在什么目标 | 对本项目的价值与限制 |
|---|---|---|
| Rush Hour／[复杂度与规则建模论文](https://air.uniud.it/retrieve/22b19a17-19a9-4b13-a1e2-3b9126cce139/paper_29.pdf)、[公开代码](https://github.com/fogleman/rush) | 可变棋盘推广版的目标可达性为PSPACE完全；车只能沿自身轴线移动 | 优先考虑。动作可逆，同一车辆布局的可达分量内有对称最短路；适合固定出口目标，也可另设完整排列目标考察任意状态对。需要区分两种目标及单格／多格动作口径 |
| Token Swapping／[ESA2016](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2016.66)、[树上的困难性](https://arxiv.org/abs/2103.06707)、[公开实例及适配](token_swapping/data/README.md) | 一步交换图上相邻节点的棋子，最少交换次数NP-hard，即使底图是树 | 最适合直接检验任意起终点的距离。连通底图上所有排列互达；仅要求最终完成时可由通用流程求解，必须把成本／最优性列为目标。tket固定问题的上游序列不保证最短，本地另用精确搜索标定 |
| 冲突图渡河／[Csorba等，SIAM](https://doi.org/10.1137/080736661) | 狼羊菜推广成任意冲突图，求可完成运输的最小船容量NP-hard | 保留为过桥邻近候选。冲突关系改变可行状态与连通性，输入需有位置、船／管理者位置、冲突图与容量。一般冲突图规则尚无本地实现 |
| 多岛多手电／[原题及本地核对](bridge/README.md#变体与适用性) | 维护人员与手电分布，人员有不同目的地；已核对的具体规则没有NP-hard结论 | 保留过桥语义、精确状态规模小，但现有来源为教学任务 |

推箱子在棋盘与箱数可增长的推广下为PSPACE完全，见[文献入口](https://webdocs.cs.ualberta.ca/~games/Sokoban/papers.html)。当前Boxoban hard是固定10×10四箱题集。

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
- SOTA 组／32B 组：共享同一任务的描述与规则，一次输出完整剩余计划 JSON、无中途反馈。五任务文字已用于用户授权的纯文字基线，预算和题集见对应正式配置。
- 监督答案单列排序、Reason、Action。地图组是排序→R→A，无地图组删除排序行；当前为简短 Reason，长 CoT 尚未准备。

## 代码和权重怎么维护

`tasks/` 是日常阅读入口；公共实现放在 `grounded_llm/core/`，规则、数据适配与裁判放在 `grounded_llm/tasks/`，正式合同与运行编排放在 `experiments/`。任务独立编排 Qmap 训练与评测，SFT 使用公共训练基础，见 [Qmap 方法](QMAP_METHOD.md)和[源码结构](../README.md#代码依赖与接入)。新训练写入 Git 外的 `runs/`；确认采用的模型才放入该任务的 `Qmap/checkpoints/`，来源写入 `selection.json`。

项目进展与下一步只维护在 [状态页](../docs/PROJECT_STATUS_AND_TODO.md)。历史结果仍按原合同解释。
