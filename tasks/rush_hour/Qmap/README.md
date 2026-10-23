# 移车出库 Qmap

**熟悉布局约98%完成，未见布局54.59%，尚未达标。** 目前没有采用权重。完整比较见[三任务报告](../../../experiments/new_tasks_qmap/results/report.md)，采用状态见[selection](selection.json)。

## 输入和目标

| 内容 | 含义 |
|---|---|
| 当前状态 | 完整6×6棋盘，区分每辆车的身份与位置 |
| 指定目标 | 同一组车辆的完整目标棋盘，所有位置明确，A在最右端 |
| 合法候选 | 每辆车沿自身方向滑动任意合法距离后的全部棋盘 |
| 学习的距离 | 到指定目标的最少滑动次数；滑一格或多格都计一步 |

当前状态、目标和候选各自经过同一个编码器，得到128维向量。编码当前状态时不读取查询目标。车辆身份始终保留，不能把多辆车当作一片无区别的障碍。

## 怎么训练和检查

每个接纳的场景完整列出可达状态，计算到指定目标的最短距离。同一个任务Qmap覆盖多种布局；测试分别检查熟悉布局的新关系、未见目标和未见布局。同一车辆配置不会跨训练与测试组。

冻结地图每一步选择预测距离最小的合法后继。所有车辆精确匹配目标才算完成，各必需组至少80%，目标90%以上；另报最优率、额外步数和循环率。

最近检查发现，只改变普通车辆的字母名称也会影响表现。为此已开展输入编码对照，但尚无最终评测，不能提前计作改进。

## 配置与证据

- [基线配置](../../../experiments/new_tasks_qmap/configs/rush_hour/rush_hour_pairs.json)、[验收要求](../../../experiments/new_tasks_qmap/configs/exact_pair_acceptance.json)
- [数据范围](../data/README.md)、[实验入口](../../../experiments/new_tasks_qmap/README.md)
- [当前较好结果](../../../experiments/new_tasks_qmap/results/rush_hour/rush_hour_pairs_spatial.json)、[未完成的编码对照配置](../../../experiments/new_tasks_qmap/configs/rush_hour/rush_hour_lane_order_pairs.json)
