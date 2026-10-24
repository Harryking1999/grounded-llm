# 8-Puzzle 数据

[example.json](example.json) 是一题可达的一步说明例，[纯文字基线准备脚本](../../../experiments/task_text_baselines/src/prepare.py)按精确距离生成评测题；Qmap数据由[本任务实现](../../../grounded_llm/tasks/puzzle8/data.py)及[实验配置](../../../experiments/puzzle8_qmap/configs/)维护，SFT见[统一实验入口](../../../experiments/map_sft_baselines/README.md)。

- 状态：3×3 整数矩阵，0 是空格，1–8 是数字块；展平时按行读取。
- 决策：当前／目标棋盘、全部合法方向、本轮候选 ID 与对应真实后继。
- 监督：真实最短距离、候选到目标的距离与最优动作。Q 距离排序在选定模型后生成，当前例中留空，不用真值冒充 Q 输出。
- 模型文字只显示当前状态、实际历史与候选方向；真实后继由环境供地图编码，不将精确距离作为推理文字。

现有[环境](../../../experiments/task_text_baselines/src/puzzle8.py)使用空格交换、边界检查和固定目标反向 BFS；评测题参考解已通过回放。后续训练样本数、关系留出与任意目标条件仍需正式合同。随机打乱步数不当作最短距离。原始数据和 oracle 表放 Git 外。
