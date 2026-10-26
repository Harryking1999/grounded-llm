# 棋子交换 Qmap

**原固定树地图通过完整完成率验收并采用。** 三个1024题组完成率97.46%、98.24%、97.46%；最优完成约71%–75%，仍有残余循环。另一棵10顶点树也有独立采用地图；两树追加效率验收未通过，优化已暂停。完整比较见[棋子交换报告](../../../experiments/new_tasks_qmap/results/token_swapping/report.md)，采用状态见[selection](selection.json)。

## 输入和目标

| 内容 | 含义 |
|---|---|
| 固定环境 | [一棵10顶点树](../data/tree10.json)，节点与棋子均编号0至9 |
| 当前状态 | 长度10的排列，第v项表示顶点v上的棋子 |
| 指定目标 | 另一个完整排列，不限于按编号归位 |
| 合法候选 | 沿九条树边分别交换一次，得到九个后继排列 |
| 学习的距离 | 两个排列之间的最少交换次数 |

排列编码保留“哪个顶点上是哪枚棋子”。当前、目标和候选各自得到128维向量，编码当前状态时不读取目标。

每张地图绑定一棵固定底图。相同排列换到另一棵树上，距离和动作会变化；两棵树分别训练的地图均通过完成率验收。12–15顶点暂未开展。

## 怎么训练和检查

全部3,628,800种排列的精确距离表已生成，任意状态对均可查到最少交换次数。它只用于监督和评测；地图选择动作时不能查询它。

分别留出起终点关系、起点状态以及两端状态。冻结地图逐步选择预测距离最小的后继，完整匹配目标才算成功。各必需组至少80%，目标90%以上；同时统计最优率、额外交换数和循环率。

增加近距离样本和强化动作排序后，完整完成率已有改善。但随机近距题容易通过，完整到达仍是主要验收标准。

## 配置与证据

- [基线配置](../../../experiments/new_tasks_qmap/archive/configs/token_swapping/token_swapping_pairs.json)、[验收要求](../../../experiments/new_tasks_qmap/configs/exact_pair_acceptance.json)
- [数据来源](../data/README.md)、[实验入口](../../../experiments/new_tasks_qmap/README.md)
- [采用地图验收](../../../experiments/new_tasks_qmap/results/token_swapping/token_endpoint_acceptance.json)、[采用配置](../../../experiments/new_tasks_qmap/configs/token_swapping/token_endpoint_l1_fit.json)
- [历史较好结果](../../../experiments/new_tasks_qmap/archive/results/token_swapping/token_pairs_local_walk8_rank4.json)、[对应配置](../../../experiments/new_tasks_qmap/archive/configs/token_swapping/token_swapping_local_walk8_rank4_pairs.json)
