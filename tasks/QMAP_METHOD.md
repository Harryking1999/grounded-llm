# Qmap：共享方法与任务外壳

统一目标是让与目标无关的 Q(s) 通过距离与候选排序监督形成状态地图。

| 层次 | 共享实现／约定 | 任务负责的部分 |
|---|---|---|
| 环境 | 返回真实状态、合法动作、真实后继、终止条件 | 状态表示与规则 |
| Q 编码器 | [StateEncoder](../grounded_llm/core/qmap/model.py)：两层隐藏 MLP、ReLU、128 维 Q；隐藏宽度由已有正式合同指定 | 输入编码与宽度；魔方已有对称呈现适配 |
| 关系学习 | 同文件的固定 `distance` 与 `soft_rank_loss`；距离 Huber＋候选排序，只训练 Q | 精确标签采样、可达性、固定距离规则 |
| 训练入口 | 复用既有训练实现，见下表 | 正式合同与对应数据 |
| 冻结读取 | [统一状态槽位](../grounded_llm/core/reader/state_q.py)、[MapBatch](../grounded_llm/core/reader/memory.py) | 将当前／目标／合法后继编码为同格式槽位 |
| LLM 选择 | 共用投影与 cross-attention 读取，再输出本轮动作 | 任务文字与动作解析 |

| 任务 | 正式训练实现入口 | 适配／加载 |
|---|---|---|
| 积木 | [多棋盘训练](../experiments/blocks_distance_map/src/multiboard_run.py)、[树关系](../experiments/blocks_distance_map/src/tree_supervision_run.py)、[续训](../experiments/blocks_distance_map/src/tree_continue_run.py)、[12块多初盘完整图联合续训](../experiments/blocks_distance_map/src/full_graph_run.py) | [BoardEncoder](../grounded_llm/tasks/blocks/model.py)、[FrozenBoardMap](../grounded_llm/tasks/blocks/qmap.py) |
| 寻路 | [run.py](../experiments/path_qmap/src/run.py) | [reader.py](../grounded_llm/tasks/path/reader.py) |
| 汉诺塔 | [汉诺塔独立训练](../experiments/hanoi_qmap/src/run.py)、[数据](../grounded_llm/tasks/hanoi/data.py)、[评测](../grounded_llm/tasks/hanoi/evaluate.py) | [环境](../grounded_llm/tasks/hanoi/environment.py)、[本任务 QMap／MapReader](../grounded_llm/tasks/hanoi/model.py) |
| 魔方 | [魔方独立训练](../experiments/cube2_qmap/src/run.py)、[关系数据](../grounded_llm/tasks/cube2/relation_data.py)、[评测](../grounded_llm/tasks/cube2/evaluate.py)；已采用模型的[循环修复续训](../experiments/cube2_qmap/src/cycle_repair.py) | [环境](../grounded_llm/tasks/cube2/environment.py)、[本任务 QMap／MapReader](../grounded_llm/tasks/cube2/model.py) |
| 8-Puzzle | [独立训练](../experiments/puzzle8_qmap/src/run.py)、[完整状态与目标关系数据](../grounded_llm/tasks/puzzle8/data.py)、[评测](../grounded_llm/tasks/puzzle8/evaluate.py) | [环境](../grounded_llm/tasks/puzzle8/environment.py)、[QMap／MapReader](../grounded_llm/tasks/puzzle8/model.py)；不复用其他任务权重 |
| Boxoban hard完整目标 | [完整状态对入口](../experiments/new_tasks_qmap/src/relations.py)、[任务配置](../experiments/new_tasks_qmap/configs/sokoban/sokoban_pairs.json) | [任务MapReader](../grounded_llm/tasks/sokoban/qmap.py)；尚未通过采用验收 |
| Rush Hour完整目标 | [完整状态对入口](../experiments/new_tasks_qmap/src/relations.py)、[任务配置](../experiments/new_tasks_qmap/configs/rush_hour/rush_hour_pairs.json) | [任务MapReader](../grounded_llm/tasks/rush_hour/qmap.py)；尚未通过采用验收 |
| Token Swapping固定树任意目标 | [完整状态对入口](../experiments/new_tasks_qmap/src/relations.py)、[任务配置](../experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json) | [任务MapReader](../grounded_llm/tasks/token_swapping/qmap.py)；尚未通过采用验收 |

当前基线复用距离Huber＋候选软排序；模型、采样、权重与距离几何以对应配置为准。改进先与现有基线比较；更换损失或监督属于单独研究条件，需明确提出，不能在实现修复中悄悄替换。积木有向距离、汉诺塔对称L1、魔方L∞和循环修复保留各自身份，不描述为完全相同的训练流程。

历史新任务的[固定目标可行性试验](../experiments/new_tasks_qmap/README.md)按任务独立调用同一小型诊断入口，不改动既有训练器。过桥比较即时耗时＋后继距离；推箱子使用成功状态集合的目标条件参照；24点主要检验可解性。不可达保留真值无穷、训练用明确的有限代理值。三者尚无采用权重，也未验证任意目标几何。

## 保留的新任务范围

以下为已实现完整状态对数据与训练入口的范围，尚无通过验收的对应地图，场景内的状态与合法动作受环境约束，跨场景泛化需把场景信息编码进输入并整组留出测试。

| 任务 | 输入与共享范围 | 监督关系与泛化 |
|---|---|---|
| [推箱子](sokoban/Qmap/README.md) | 10×10四通道棋盘：墙、目标、箱子、玩家；一个Qmap覆盖多张Boxoban hard布局 | 每题指定完整成功目标，玩家和箱子均匹配；最少逐格走／推次数；墙体等价类整组留出 |
| [Rush Hour](rush_hour/Qmap/README.md) | 无墙6×6按车辆身份分通道的完整棋盘；一个Qmap覆盖多种布局 | 每题指定完整成功目标，所有车均匹配；最少滑动次数；车辆配置及可达分量整组留出 |
| [Token Swapping](token_swapping/Qmap/README.md) | 固定公开10顶点树，完整排列的顶点×棋子one-hot | 任意完整起终点的最少边交换次数；关系与反向关系成组留出；验收后再考虑12–15顶点 |

三项GOAL槽位都放真实完整状态，各题只有一个目标，Q(s)与Q(g)独立编码；保持现有cross-attention读取当前、单目标与候选的条件，不增加多目标聚合。前两项同一布局可采多个完整成功终局作为不同题目的目标，但目标域仍限于原规则成功终局，新训练与评测只采用指定目标，原Flash筛查仅为历史记录。Token Swapping允许任意完整排列目标；棋子重标号只用于精确教师标签查询。

三项按任务独立编排，基线复用现有MLP、128维Q、距离Huber、soft-rank及MapBatch；编码与排序权重对照由各自配置明确。reader提供当前、一个完整目标、全部合法真实后继；先验收冻结Q本身的距离、动作和完整计划，再接公共SFT；指定目标文字与裁判已接入，训练、冻结reader与正式配置见[实验入口](../experiments/new_tasks_qmap/README.md)，复用公共编码与监督。必需组地图贪心至少80%、目标90%以上，未达标不采用。

汉诺塔和魔方分别拥有训练、数据、模型适配与评测编排，不使用混合 `puzzle` 训练器作为当前入口。基础指标和批次采样放在 `grounded_llm/core/qmap/metrics.py`、`grounded_llm/core/qmap/sampling.py`；旧 `puzzle_qmap` 仅为历史复现与导入兼容保留。寻路采用 `path_qmap` 入口与 `grounded_llm/tasks/path/` 实现，不回退旧 Q/V。

`Qmap/selection.json` 记录已经采用的 checkpoint、选择标准与适用条件。寻路采用该任务 selection 中的纯 Q。新条件没有已核实权重时保持空缺。
