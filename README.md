# Grounded LLM with Cognitive Maps

研究显式状态地图能否提升 LLM 的动作判断和连续任务完成。环境提供当前状态、合法候选与执行反馈；地图经 cross-attention 接入 LLM。

当前按寻路、积木、汉诺塔和魔方分别整理规则外壳、文字、数据和 Qmap，先针对任务独立训练与评测，再考虑泛化和合并。积木同数据多基线及 10k／7B 训练已有完整结果；下一步先统一任务文字、测试强基线并调试地图有效的条件。

## 阅读入口

1. [项目状态与有序 TODO](docs/PROJECT_STATUS_AND_TODO.md)：唯一项目进度页。
2. [研究简述](docs/RESEARCH_BRIEF.md)：假设、定义与证伪边界。
3. [按任务工作区](tasks/README.md)：每个任务的 Qmap、文字、数据与独立实现；[全部文字总览](TASK_TEXTS.md)方便集中查看。
4. [积木多基线结果](experiments/map_sft_baselines/results/report.md)、[实验目录](experiments/README.md)：已有指标、正式配置与历史证据。
5. [开发机手册](DEVELOPMENT_MACHINES.md)：连接、存储与迁移方法。

文献和任务来源见 [Related Work](RELATED_WORK.md)、[GCML Tasks](docs/GCML_TASKS.md)。协作约定见 [AGENTS.md](AGENTS.md)。

全局文档不维护局部运行时间线。实验参数放在对应 `configs/`，结果在实验报告，原始数据、模型、日志、PID 和本机凭据留在 Git 外。
