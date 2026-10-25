# 推箱子 Qmap

**当前未采用地图；主条件改为每套固定墙体独立训练Qmap。** 同墙体内仍改变箱子、玩家及完整中间目标，跨墙泛化不作为该条件的验收要求。以下跨布局成绩属于原共享地图条件。 已完成1,000张训练棋盘的覆盖量对照，在100张未见棋盘测试中完整贪心完成率为37.89%；21步及以上0/312。结果与问题见[三任务报告](../../../experiments/new_tasks_qmap/results/report.md)，采用状态见[selection](selection.json)。

同目标下连续5–10步完整局部路径的三组对照已完成固定验证：原采样、路径打散与路径成组的未见布局到达率分别为36.52%、34.67%、35.94%，21步以上仍未成功。需区分训练关系拟合不足与跨布局泛化不足；详见[配对验证](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_local_path_validation.json)。

## 输入和目标

| 内容 | 含义 |
|---|---|
| 当前状态 | 完整10×10棋盘，保留墙、目标格、四个箱子和玩家 |
| 指定目标 | 同一布局的一张完整棋盘，箱子已到位，玩家终点也明确 |
| 合法候选 | 玩家下一步走动或推动后的全部合法棋盘，最多四个 |
| 学习的距离 | 从当前棋盘到指定目标的最少走／推步数；每次都计一步 |

当前状态、目标和候选各自经过同一个编码器，得到128维向量。编码当前状态时不读取查询目标。每张新地图对应固定墙体方向与原始目标格标记，查询目标由目标状态另行指定；MapReader核对其适用棋盘。

## 怎么训练和检查

训练使用精确求解得到的距离，同时学习比较各个候选离目标的远近。推箱子可能有去无回，距离要保留方向性。合法但会导致目标不可达的动作也保留，让地图学习区分这些分支。

**固定墙体条件分别验收留出关系、留出完整目标与原题起点执行。** 原题起点到训练目标属于拟合检查，到留出目标属于查询泛化；两者分开报告。完整合法候选保留，训练关系及其候选闭包不得泄漏保护查询；新墙体使用另训地图。原共享条件仍保留整墙体等价类留出的历史证据。

最终验收让冻结地图逐步贪心选动作，到达完整目标才成功。各必需组至少80%，目标90%以上；同时记录最优率、循环和进入死锁的比例。环境负责给出合法后继，地图负责比较距离。

## 配置与证据

- 固定墙体：[902](../../../experiments/new_tasks_qmap/configs/sokoban/sokoban_fixed_wall_902.json)、[327](../../../experiments/new_tasks_qmap/configs/sokoban/sokoban_fixed_wall_327.json)、[验收](../../../experiments/new_tasks_qmap/configs/sokoban/sokoban_fixed_wall_acceptance.json)
- [基线配置](../../../experiments/new_tasks_qmap/configs/sokoban/sokoban_pairs.json)、[验收要求](../../../experiments/new_tasks_qmap/configs/exact_pair_acceptance.json)
- [数据范围](../data/README.md)、[实验入口](../../../experiments/new_tasks_qmap/README.md)
- [当前覆盖量对照](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_coverage_comparison.json)、[采样对照](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_pairs_balanced_relations.json)
