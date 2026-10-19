# 积木 Qmap

当前采用清空候选 **835/1000（83.5%）**，正式验收须≥900/1000且历史510无明显退化；**目前未完成**。原810、789及历史权重保留；当前来源和权重见[selection.json](selection.json)。

正式目标仅为12块原八形状积木清空，使用[固定1000题](../data/clear1000.json)及[验收合同](../../../experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json)。历史510为424/510，独立列出；非空及近清空目标仅辅助，不能冒充清空成绩。

原两层MLP、128维Q、Huber、soft-rank、Adam与纯贪心不变，参见[统一方法](../../QMAP_METHOD.md)。仅沿用已授权不可达19。训练模块`experiments.blocks_distance_map.src.full_graph_run`，固定验收模块`blocks12_eval`；续训显式传入正式合同中的`continuation_checkpoint`作`--resume`，兼容原`FrozenBoardMap`加载。

[简明报告](../../../experiments/blocks_distance_map/results/blocks12_joint_report.md)与[能力总表](../../README.md#任务能力总表)维护当前成绩；过程断点、日志和曝光留在忽略runtime。当前地图的文字40题、1.5B／7B接图成绩未测，不沿用旧权重分数。
