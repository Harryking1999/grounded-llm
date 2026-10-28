# 8-Puzzle Qmap

独立训练入口为 `experiments/puzzle8_qmap/src/run.py`，正式合同为 `experiments/puzzle8_qmap/configs/puzzle8.json`，方法和运行结果见[实验 README](../../../experiments/puzzle8_qmap/README.md)。

与其他 Qmap 共享两层 MLP、128 维 Q、固定对称距离及距离／候选排序监督。输入为九位置的九类独热。完整覆盖全状态到复原终点，并构造全状态到多组中间目标的精确距离；最终以复原为主要 rollout，留出中间目标检查关系补全。

原复原权重为 `checkpoints/selected_98_28.pt`。目标规范化条件的已验证权重为 `checkpoints/goal_normalized_290000.pt`：实际换目标1009/1024（98.54%），角、边、中央三组均超过90%；原固定中间目标测试1015/1024（99.12%）。权重与读取条件由[selection.json](selection.json)维护，分项比较见[当前报告](../../../experiments/puzzle8_qmap/results/report.md)。

目标坐标读取通过 `MapReader(env, checkpoint, coordinate_frame="goal_normalized")` 启用，使用已知棋盘对称及数字重命名；默认绝对状态编码保持原条件。比较及机制见[实验报告](../../../experiments/puzzle8_qmap/results/report.md)。

后续新建的测试与训练可采用 `goal_normalized` 及selection中的对应权重，当前、目标和全部合法后继在同一目标坐标系中读取。当前4／5／6类FLAN实验使用已冻结的旧地图数据，仍为原绝对状态编码；其中5／6类训练华容道，4类将其作为整类OOD。配套评测沿用该批训练的旧地图条件，见[FLAN说明](../../../experiments/map_sft_baselines/README.md#训练什么给模型看什么)。

同一批Flan题、32步预算下，旧绝对条件到达ID 124/125、小OOD 21/125；新规范化条件为124/125、123/125。新条件的地图贪心已验收，但现有SFT入口仍直接 `encode(states)`，需接入全部槽位的目标坐标并重建匹配排序缓存，不能仅替换权重。证据见[当前报告](../../../experiments/puzzle8_qmap/results/report.md)。
