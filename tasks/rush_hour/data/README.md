# Rush Hour数据

来源为[公开题库预览](https://www.michaelfogleman.com/static/rush/rush1000.txt)，[格式定义](https://www.michaelfogleman.com/rush/)包含最短move数、36字符棋盘及可达分量大小。数据URL未提供版本号；实际使用的原始文件和冻结题目保留在实验配置指定的缓存及运行目录，不用代码提交号冒充数据文件版本。

[准备入口](../../../experiments/task_text_baselines/src/prepare_rush_token.py)按[合同](../../../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)选无墙原题。每个分量完整BFS核对源距离与状态数；原题及最短轨迹上的中间状态组成不同距离起点。相同分量内样本相关，不能当作独立原题或未见布局泛化；未据此训练地图。参考解和精确距离不进入Flash提示。

动作是`{"vehicle":"B","displacement":-2}`。横车负为左、正为右，竖车负为上、正为下，途中不能穿车／墙。`o`与`.`都作空格；渲染提示统一用`.`。目标不是移出棋盘，而是A占据所在行最右两格；其他车辆位置不限。逐格移动步数不是本任务成本。

[example.json](example.json)仅说明格式，不作为评测题或few-shot。
