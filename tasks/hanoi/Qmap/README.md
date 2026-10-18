# 汉诺塔 Qmap

训练与加载：`experiments/hanoi_qmap/src/run.py`、`data.py`、`model.py`、`evaluate.py`；本任务独立实现，不再通过混合 puzzle 训练入口。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。新时间戳不等于更好的权重，不用失败续训覆盖已采用模型。

当前采用[八盘全图联合训练](../../../experiments/hanoi_qmap/configs/hanoi8_full_coverage.json)所选权重，文件为[checkpoints/hanoi8_full_coverage_fb7ffe6_best.pt](checkpoints/hanoi8_full_coverage_fb7ffe6_best.pt)。正式[评测范围](../../../experiments/hanoi_qmap/configs/hanoi8_tower_evaluation.json)是任意合法起点到A、B或C柱上的整塔终点；不是任意中间目标求解。结果见[实验报告](../../../experiments/hanoi_qmap/results/report.md)。

加载入口为`experiments.hanoi_qmap.src.model.MapReader`，环境使用`Hanoi(8)`。读取`selection.json`中的`checkpoint`时，以本目录为相对路径基准；模型参数和动作制由断点校验，推理只使用状态特征与合法候选，不读取精确距离表。六盘参考`checkpoints/hanoi6_best.pt`继续保留，来源在`historical_reference`，不作为当前八盘权重。
