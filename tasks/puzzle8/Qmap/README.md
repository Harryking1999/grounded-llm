# 8-Puzzle Qmap

独立训练入口为 `experiments/puzzle8_qmap/src/run.py`，正式合同为 `experiments/puzzle8_qmap/configs/puzzle8.json`，方法和运行结果见[实验 README](../../../experiments/puzzle8_qmap/README.md)。

与其他 Qmap 共享两层 MLP、128 维 Q、固定对称距离及距离／候选排序监督。输入为九位置的九类独热。完整覆盖全状态到复原终点，并构造全状态到多组中间目标的精确距离；最终以复原为主要 rollout，留出中间目标检查关系补全。

用户已选用全状态复原98.28%的权重，入口见 [selection.json](selection.json)，本地推理文件为 `checkpoints/selected_98_28.pt`。固定中间目标328/1024，严格中间关系保持门槛尚未通过；具体能力范围见[当前报告](../../../experiments/puzzle8_qmap/results/report.md)。各任务后续安排以项目状态页为准。
