# 二阶魔方 Qmap

训练与加载：`experiments/cube2_qmap/src/run.py`、`relation_data.py`、`model.py`、`evaluate.py`；当前采用权重的续训入口为同目录 `cycle_repair.py`，复原为 `fixed_goal.py`。不调用汉诺塔训练器。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。新时间戳不等于更好的权重，不用失败续训覆盖已采用模型。

当前采用循环修复对照的 `repair_guard` 复原权重，保留对称 Q 与 L∞。尚未取得可靠的全局长程与任意目标结果；新一轮文字条件先按二阶整理。
