# 棋子交换数据

公开来源为[Quantinuum/tket固定测试题](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/Data/FixedCompleteSolutions.cpp)。它提供图、起始排列和参考交换序列；[上游说明](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/Data/FixedCompleteSolutions.hpp)不保证序列最短。本项目独立求最优距离。

## 当前数据

第一版固定使用公开的“Medium random tree 1”，完整边表见[tree10.json](tree10.json)。十个顶点上共有3,628,800种排列，任意两种都能互达；最远需要25次交换。

地图训练使用多种起点和目标，不只学习按编号归位。划分分别检验未见关系、未见起点和两端都未见，留出关系也不会通过候选排序进入训练。实际规模见[数据统计](../../../experiments/new_tasks_qmap/results/shared/exact_pairs_data.json)，参数见[配置](../../../experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json)。

全状态距离表用于生成正确标签，不等于全部状态对都已训练。训练样本、固定测试题和模型分别保存。

## 评测范围

完成指所有棋子与目标一致；最优指交换次数也达到精确最小值。另记录成功解多用了多少步，失败题仍计入整体完成率和最优率的分母。

早期DeepSeek的40题来自八种图，其中五题使用当前这棵树。指定目标的五题诊断是已知问题的等价重标，不能作为新的总体抽样。见[文字结果](../../../experiments/task_text_baselines/results/exact_goals_flash64k.json)。

当前未训练其他树或12–15顶点条件。[example.json](example.json)仅说明格式；源数据按上游Apache-2.0许可保留。
