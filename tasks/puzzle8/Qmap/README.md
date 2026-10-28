# 8-Puzzle Qmap

独立训练入口为 `experiments/puzzle8_qmap/src/run.py`，正式合同为 `experiments/puzzle8_qmap/configs/puzzle8.json`，方法和运行结果见[实验 README](../../../experiments/puzzle8_qmap/README.md)。

与其他 Qmap 共享两层 MLP、128 维 Q、固定对称距离及距离／候选排序监督。输入为九位置的九类独热。完整覆盖全状态到复原终点，并构造全状态到多组中间目标的精确距离；最终以复原为主要 rollout，留出中间目标检查关系补全。

原复原权重为 `checkpoints/selected_98_28.pt`。目标规范化条件的已验证权重为 `checkpoints/goal_normalized_290000.pt`：实际换目标1009/1024（98.54%），角、边、中央三组均超过90%；原固定中间目标测试1015/1024（99.12%）。权重与读取条件由[selection.json](selection.json)维护，分项比较见[当前报告](../../../experiments/puzzle8_qmap/results/report.md)。

目标坐标读取通过 `MapReader(env, checkpoint, coordinate_frame="goal_normalized")` 启用，使用已知棋盘对称及数字重命名；默认绝对状态编码保持原条件。比较及机制见[实验报告](../../../experiments/puzzle8_qmap/results/report.md)。
