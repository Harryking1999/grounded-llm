# Qmap：共享方法与任务外壳

统一目标是让与目标无关的 Q(s) 通过距离与候选排序监督形成状态地图。不是为每个任务手写一套决策策略，也不是把不同任务训练出的向量当成已经对齐的坐标。

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

当前所有任务保持统一的距离 Huber＋候选软排序监督和损失配方；针对失败，先从数据覆盖、关系组合与采样修正，不新增任务专属损失或监督。该约束适用于当前及后续工作。历史运行并非全部具有相同采样、损失权重或距离几何。积木有向距离、汉诺塔对称 L1、当前魔方 L∞ 的区别保留在配置中。魔方的额外循环修复属于明确记录的任务适配，不将其描述为所有任务已有的相同训练流程。

新任务的[固定目标可行性试验](../experiments/new_tasks_qmap/README.md)按任务独立调用同一小型诊断入口，不改动既有训练器。过桥比较即时耗时＋后继距离；推箱子使用成功状态集合的目标条件参照；24点主要检验可解性。不可达保留真值无穷、训练用明确的有限代理值。三者尚无采用权重，也未验证任意目标几何。

汉诺塔和魔方分别拥有训练、数据、模型适配与评测编排，不使用混合 `puzzle` 训练器作为当前入口。基础指标和批次采样放在 `grounded_llm/core/qmap/metrics.py`、`grounded_llm/core/qmap/sampling.py`；旧 `puzzle_qmap` 仅为历史复现与导入兼容保留。寻路采用 `path_qmap` 入口与 `grounded_llm/tasks/path/` 实现，不回退旧 Q/V。

`Qmap/selection.json` 记录已经采用的 checkpoint、选择标准与适用条件。魔方的复原断点不能称为任意目标最优地图；汉诺塔六盘历史权重不能用于八盘；寻路采用该任务 selection 中的纯 Q。新条件没有已核实权重时保持空缺，不能用旧任务或旧动作制权重替代。
