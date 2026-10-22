# Rush Hour数据

来源为[公开题库预览](https://www.michaelfogleman.com/static/rush/rush1000.txt)，[格式定义](https://www.michaelfogleman.com/rush/)包含最短move数、36字符棋盘及可达分量大小。数据URL未提供版本号；实际使用的原始文件和冻结题目保留在实验配置指定的缓存及运行目录，不用代码提交号冒充数据文件版本。

[准备入口](../../../experiments/task_text_baselines/src/prepare_rush_token.py)按[合同](../../../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)选无墙原题。每个分量完整BFS核对源距离与状态数；原题及最短轨迹上的中间状态组成不同距离起点。相同分量内样本相关，不能当作独立原题或未见布局泛化；未据此训练地图。参考解和精确距离不进入Flash提示。

动作是`{"vehicle":"B","displacement":-2}`。横车负为左、正为右，竖车负为上、正为下，途中不能穿车／墙。`o`与`.`都作空格；渲染提示统一用`.`。目标不是移出棋盘，而是A占据所在行最右两格；其他车辆位置不限。逐格移动步数不是本任务成本。

[example.json](example.json)仅说明格式，不作为评测题或few-shot。

## 多布局Qmap数据

首版已确认同一Qmap覆盖多种无墙6×6布局，每题给出一个所有车位置都明确、A到最右端的完整成功目标。每个场景保留静态车辆目录、原题ID及实际起终点棋盘；完整枚举接纳分量，对选定目标分别单源BFS标注最少滑动次数，同场景采多个目标样本。只评指定目标。现有8个分量共54,171个状态，只是已核对的开发诊断，完整状态对训练池及整组留出已按[正式配置](../../../experiments/new_tasks_qmap/configs/rush_hour_pairs.json)生成，实际图与关系统计见[数据证据](../../../experiments/new_tasks_qmap/results/exact_pairs_data.json)。

按整个静态车辆配置及分量分组，不把同一分量的原题与轨迹中间状态拆到训练和测试。相同配置的其他分量及仅重命名普通车的副本也留在同组。独立最终测试需另留未曝光布局，场景范围及采样预算唯一维护在正式配置；不把已有40起点当作40个独立布局。编码、目标条件与成本验收见[Qmap说明](../Qmap/README.md)。
