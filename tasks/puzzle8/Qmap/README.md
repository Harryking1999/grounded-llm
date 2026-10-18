# 8-Puzzle Qmap

独立训练入口为 `experiments/puzzle8_qmap/src/run.py`，正式合同为 `experiments/puzzle8_qmap/configs/puzzle8.json`，方法和运行结果见[实验 README](../../../experiments/puzzle8_qmap/README.md)。

与其他 Qmap 共享两层 MLP、128 维 Q、固定对称距离及距离／候选排序监督。输入为九位置的九类独热。完整覆盖全状态到复原终点，并构造全状态到多组中间目标的精确距离；最终以复原为主要 rollout，留出中间目标检查关系补全。覆盖全部状态不代表全部状态对或未见状态泛化。

尚无已验证采用权重；选定后放入 Git 忽略的 `checkpoints/` 并更新 [selection.json](selection.json)。不使用其他任务或旧 XoT 权重替代。
