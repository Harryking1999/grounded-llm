# 棋子交换 Qmap

**当前主要测试组完成率约68%–69%，尚未达标。** 仍有约三成题循环，尚无采用权重。完整比较见[三任务报告](../../../experiments/new_tasks_qmap/results/report.md)，采用状态见[selection](selection.json)。

## 输入和目标

| 内容 | 含义 |
|---|---|
| 固定环境 | [一棵10顶点树](../data/tree10.json)，节点与棋子均编号0至9 |
| 当前状态 | 长度10的排列，第v项表示顶点v上的棋子 |
| 指定目标 | 另一个完整排列，不限于按编号归位 |
| 合法候选 | 沿九条树边分别交换一次，得到九个后继排列 |
| 学习的距离 | 两个排列之间的最少交换次数 |

排列编码保留“哪个顶点上是哪枚棋子”。当前、目标和候选各自得到128维向量，编码当前状态时不读取目标。

底图固定在本任务中。相同排列换到另一棵树上，距离和动作会变化；当前实验检验的是新状态、新关系，不是跨树泛化。12–15顶点暂未开展。

## 怎么训练和检查

全部3,628,800种排列的精确距离表已生成，任意状态对均可查到最少交换次数。它只用于监督和评测；地图选择动作时不能查询它。

分别留出起终点关系、起点状态以及两端状态。冻结地图逐步选择预测距离最小的后继，完整匹配目标才算成功。各必需组至少80%，目标90%以上；同时统计最优率、额外交换数和循环率。

增加近距离样本和强化动作排序后，完整完成率已有改善。但随机近距题容易通过，不代表长轨迹中遇到的所有局面都可靠；完整到达仍是主要验收标准。

## 配置与证据

- [基线配置](../../../experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json)、[验收要求](../../../experiments/new_tasks_qmap/configs/exact_pair_acceptance.json)
- [数据来源](../data/README.md)、[实验入口](../../../experiments/new_tasks_qmap/README.md)
- [当前较好结果](../../../experiments/new_tasks_qmap/results/token_swapping/token_pairs_local_walk8_rank4.json)、[对应配置](../../../experiments/new_tasks_qmap/configs/token_swapping/token_swapping_local_walk8_rank4_pairs.json)
