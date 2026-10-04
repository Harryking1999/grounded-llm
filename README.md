# Grounded LLM with Cognitive Maps

研究显式状态模型能否帮助 LLM 理解状态、判断动作后果并完成规划。当前主实验是**单图寻路与 1000 初始棋盘积木的完整轨迹 SFT**：冻结 Qwen2.5-1.5B-Instruct 和地图，训练地图读取接口，输出远近排序、动作、终止与真实历史总结。积木回答只列最近 Top-10，输入仍给全部合法候选。

## 阅读入口

- [当前状态与有序 TODO](docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页及运行定位。
- [研究简述](docs/RESEARCH_BRIEF.md)：研究问题、信息条件与证伪标准。
- [当前实验](experiments/flamingo_map_reader/README.md)：分支、源码和文件入口。
- [训练与评测设置](experiments/flamingo_map_reader/DESIGN.md)：实际数据数量、动作覆盖、提示文本、监督与优化设置。
- [结果报告](experiments/flamingo_map_reader/results/report.md)：已收录 reference 与闭环结果，并为未完成评测保留填写位置。
- [历史结果索引](docs/PROJECT_STATUS_AND_TODO.md#历史证据入口)：显式距离接口、旧连续 token、地图训练和独立模型基线。
- [Related Work](RELATED_WORK.md)、[GCML 任务来源](docs/GCML_TASKS.md)。

## 仓库约定

正式配置、源码和紧凑结果放在 `experiments/<study>/`；原始数据、日志和模型放在 Git 忽略路径。运行分支与协议提交以实验 README 为准，旧分支的文档可能描述较早方案。协作与 Git 纪律见 [AGENTS.md](AGENTS.md)。
