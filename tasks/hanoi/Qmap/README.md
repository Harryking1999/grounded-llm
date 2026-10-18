# 汉诺塔 Qmap

训练与加载：`experiments/hanoi_qmap/src/run.py`、`data.py`、`model.py`、`evaluate.py`；本任务独立实现，不再通过混合 puzzle 训练入口。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。新时间戳不等于更好的权重，不用失败续训覆盖已采用模型。

已有三至六盘历史 Qmap；八盘完整距离、关系轨道划分和六种柱置换增强已准备，正式配置见 `experiments/hanoi_qmap/configs/hanoi8.json`。本轮结果写入实验报告；八盘尚无已验证采用权重，不能把六盘权重当作八盘地图。
