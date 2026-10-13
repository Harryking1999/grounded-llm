# Grounded LLM with Cognitive Maps

研究显式状态地图能否提升 LLM 的动作判断和连续任务完成。环境提供当前状态、合法候选与执行反馈；地图经 cross-attention 接入 LLM。

当前重点是积木的**同数据多基线对照**：统一示范和损失，比较有无地图、仅训接口与全参训练。先前冻结接口、融合 FFN 和无地图 SFT 均作为独立历史实验保留。

## 阅读入口

1. [项目状态与有序 TODO](docs/PROJECT_STATUS_AND_TODO.md)：唯一项目进度页。
2. [研究简述](docs/RESEARCH_BRIEF.md)：假设、定义与证伪边界。
3. [同数据多基线](experiments/map_sft_baselines/README.md)：设计、配置、代码和结果入口。
4. [实验目录](experiments/README.md)：既有实验及历史证据。
5. [开发机手册](DEVELOPMENT_MACHINES.md)：连接、存储与迁移方法。

文献和任务来源见 [Related Work](RELATED_WORK.md)、[GCML Tasks](docs/GCML_TASKS.md)。协作约定见 [AGENTS.md](AGENTS.md)。

全局文档不维护局部运行时间线。实验参数放在对应 `configs/`，结果在实验报告，原始数据、模型、日志、PID 和本机凭据留在 Git 外。
