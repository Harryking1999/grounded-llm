# Qmap：共享方法与任务外壳

统一目标是让与目标无关的 Q(s) 通过距离与候选排序监督形成状态地图。不是为每个任务手写一套决策策略，也不是把不同任务训练出的向量当成已经对齐的坐标。

| 层次 | 共享实现／约定 | 任务负责的部分 |
|---|---|---|
| 环境 | 返回真实状态、合法动作、真实后继、终止条件 | 状态表示与规则 |
| Q 编码器 | [StateEncoder](../experiments/blocks_distance_map/src/model.py)：两层隐藏 MLP、ReLU、128 维 Q；隐藏宽度由已有正式合同指定 | 输入编码与宽度；魔方已有对称呈现适配 |
| 关系学习 | 同文件的固定 `distance` 与 `soft_rank_loss`；距离 Huber＋候选排序，只训练 Q | 精确标签采样、可达性、固定距离规则 |
| 训练入口 | 复用既有训练实现，见下表 | 正式合同与对应数据 |
| 冻结读取 | [统一状态槽位](../experiments/flamingo_map_reader/src/state_q.py)、[MapBatch](../experiments/flamingo_map_reader/src/memory.py) | 将当前／目标／合法后继编码为同格式槽位 |
| LLM 选择 | 共用投影与 cross-attention 读取，再输出本轮动作 | 任务文字与动作解析 |

| 任务 | 正式训练实现入口 | 适配／加载 |
|---|---|---|
| 积木 | [多棋盘训练](../experiments/blocks_distance_map/src/multiboard_run.py)、[树关系](../experiments/blocks_distance_map/src/tree_supervision_run.py)、[续训](../experiments/blocks_distance_map/src/tree_continue_run.py) | [BoardEncoder](../experiments/blocks_distance_map/src/model.py)、[FrozenBoardMap](../experiments/flamingo_map_reader/src/blocks.py) |
| 寻路 | [run.py](../experiments/path_qmap/src/run.py) | [reader.py](../experiments/path_qmap/src/reader.py) |
| 汉诺塔 | [汉诺塔独立训练](../experiments/hanoi_qmap/src/run.py)、[数据](../experiments/hanoi_qmap/src/data.py)、[评测](../experiments/hanoi_qmap/src/evaluate.py) | [环境](../experiments/hanoi_qmap/src/environment.py)、[本任务 QMap／MapReader](../experiments/hanoi_qmap/src/model.py) |
| 魔方 | [魔方独立训练](../experiments/cube2_qmap/src/run.py)、[关系数据](../experiments/cube2_qmap/src/relation_data.py)、[评测](../experiments/cube2_qmap/src/evaluate.py)；已采用模型的[循环修复续训](../experiments/cube2_qmap/src/cycle_repair.py) | [环境](../experiments/cube2_qmap/src/environment.py)、[本任务 QMap／MapReader](../experiments/cube2_qmap/src/model.py) |
| 8-Puzzle | 尚未实现或训练 | [任务适配规划](puzzle8/Qmap/README.md)；不复用其他任务权重 |

“方法对齐”指共享模型骨干、关系监督和接口；不表示所有历史运行具有相同采样、损失权重或距离几何。积木有向距离、汉诺塔对称 L1、当前魔方 L∞ 的区别保留在配置中。魔方的额外循环修复属于明确记录的任务适配，不将其描述为所有任务已有的相同训练流程。

汉诺塔和魔方分别拥有训练、数据、模型适配与评测编排，不使用混合 `puzzle` 训练器作为当前入口。基础指标和批次采样放在 `grounded_llm/qmap_metrics.py`、`grounded_llm/qmap_sampling.py`；旧 `puzzle_qmap` 仅为历史复现与导入兼容保留。寻路采用同一分支新增的 `path_qmap` 实现，不回退旧 Q/V。

`Qmap/selection.json` 记录已经采用的 checkpoint。`best` 必须同时说明选择标准与适用条件；魔方是用户指定的复原断点，不能称为任意目标最优地图。汉诺塔六盘历史权重不能用于声称已完成八盘。寻路新纯 Q 没有已核实的成功权重时保持空缺，不能用旧 Q/V 替代。
