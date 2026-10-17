# 积木 Qmap

训练与加载：`experiments/blocks_distance_map/src/tree_continue_run.py`；加载 `FrozenBoardMap`。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。新时间戳不等于更好的权重，不用失败续训覆盖已采用模型。

现有积木同数据多基线已完成，结果及 B2_10k 异常诊断见原实验报告。任务文字沿用本轮真实实现；新的难度与正式强基线预算待确认。
