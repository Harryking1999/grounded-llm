# 积木 Qmap

当前采用用户指定的 **ffba71f结构化1000步候选**，正式优化与验收为12块清空，固定1000题781/1000（78.1%）。权重为 `checkpoints/structured_1000_ffba71f.pt`；原选用权重和首轮421父模型保留。

训练与评测：`experiments.blocks_distance_map.src.full_graph_run`、`full_graph_eval`；兼容原 `FrozenBoardMap` 加载。历史训练入口为 `tree_continue_run.py`。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。新时间戳不等于更好的权重，不用失败续训覆盖已采用模型。

成绩及比较范围见[能力总表](../../README.md#任务能力总表)。[固定1000题](../data/clear1000.json)与[验收合同](../../../experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json)用于正式同题比较；原510回归421/510单列历史指标。非空目标仅作辅助诊断，地标与完整分支仍参与联合训练。当前权重的1.5B接图训练尚无对应结果，历史SFT成绩单独标记。
