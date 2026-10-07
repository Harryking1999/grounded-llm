# Grounded LLM with Cognitive Maps

研究 LLM 能否通过 **cross-attention 读取并利用显式状态地图**，改善动作选择与长程任务完成。当前实验冻结 Qwen2.5-1.5B-Instruct 和地图，只训练读取接口。寻路沿用地址 K 与状态 V 分离的方案；积木本次加入融合 FFN 和失败上下文重新训练，答案限 Top-10，输入保留全部候选。

## 阅读入口

1. [训练与评测设计](experiments/flamingo_map_reader/DESIGN.md)：数据、文本、监督与指标。
2. [本轮结果](experiments/flamingo_map_reader/results/report.md)：大 batch 负结果与小 batch 证据状态；[上一轮报告](experiments/flamingo_map_reader/results/report_long_trajectory.md)保留原两任务成绩与地图使用证据。
3. [当前状态与 TODO](docs/PROJECT_STATUS_AND_TODO.md)：唯一进度页及运行定位。
4. [研究简述](docs/RESEARCH_BRIEF.md)：问题、主张与证据边界。

实现和运行入口见[当前实验 README](experiments/flamingo_map_reader/README.md)。地图依赖及独立基线见[实验目录](experiments/README.md)；文献与任务来源见 [Related Work](RELATED_WORK.md)、[GCML_TASKS](docs/GCML_TASKS.md)。旧变体见[历史索引](docs/PROJECT_STATUS_AND_TODO.md#历史证据入口)。

源码、正式配置与结果摘要纳入 Git；原始数据、模型、日志及机器连接信息留在忽略路径。协作约定见 [AGENTS.md](AGENTS.md)。
