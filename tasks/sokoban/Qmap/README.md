# 推箱子 Qmap

**当前还不能可靠求解新棋盘。** 已完成1,000张训练棋盘的覆盖量对照，在100张未见棋盘测试中完整贪心完成率为37.89%；21步及以上0/312。结果与问题见[三任务报告](../../../experiments/new_tasks_qmap/results/report.md)，采用状态见[selection](selection.json)。

同目标下连续5–10步完整局部路径的三组对照已完成固定验证：原采样、路径打散与路径成组的未见布局到达率分别为36.52%、34.67%、35.94%，21步以上仍未成功。需区分训练关系拟合不足与跨布局泛化不足；详见[配对验证](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_local_path_validation.json)。

## 输入和目标

| 内容 | 含义 |
|---|---|
| 当前状态 | 完整10×10棋盘，保留墙、目标格、四个箱子和玩家 |
| 指定目标 | 同一布局的一张完整棋盘，箱子已到位，玩家终点也明确 |
| 合法候选 | 玩家下一步走动或推动后的全部合法棋盘，最多四个 |
| 学习的距离 | 从当前棋盘到指定目标的最少走／推步数；每次都计一步 |

当前状态、目标和候选各自经过同一个编码器，得到128维向量。编码当前状态时不读取查询目标。同一张任务地图覆盖多种布局，墙和目标格随棋盘一并输入。

## 怎么训练和检查

训练使用精确求解得到的距离，同时学习比较各个候选离目标的远近。推箱子可能有去无回，距离要保留方向性。合法但会导致目标不可达的动作也保留，让地图学习区分这些分支。

**以未见布局泛化为核心验收。** 训练布局的新起终点关系仅用于诊断拟合程度，另查未见目标，相同墙体及其旋转、镜像版本划在同一组，训练覆盖同时考察棋盘、状态与目标组合、合法动作分支和连续路径。

最终验收让冻结地图逐步贪心选动作，到达完整目标才成功。各必需组至少80%，目标90%以上；同时记录最优率、循环和进入死锁的比例。环境负责给出合法后继，地图负责比较距离。

## 配置与证据

- [基线配置](../../../experiments/new_tasks_qmap/configs/sokoban/sokoban_pairs.json)、[验收要求](../../../experiments/new_tasks_qmap/configs/exact_pair_acceptance.json)
- [数据范围](../data/README.md)、[实验入口](../../../experiments/new_tasks_qmap/README.md)
- [当前覆盖量对照](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_coverage_comparison.json)、[采样对照](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_pairs_balanced_relations.json)
