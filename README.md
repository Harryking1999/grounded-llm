# Grounded LLM with Cognitive Maps

研究显式状态地图能否改善 LLM 的动作判断和连续任务完成。当前环境提供状态、合法候选与执行反馈，地图通过 cross-attention 接入 LLM；研究范围与证据边界见[研究简述](docs/RESEARCH_BRIEF.md)。

## 日常入口

- **现在做什么：**[项目状态与有序 TODO](docs/PROJECT_STATUS_AND_TODO.md)。
- **某个任务怎么做、做到什么程度：**[五任务工作区与能力表](tasks/README.md)。
- **代码在哪里、哪些可以复用：**[实验与实现索引](experiments/README.md)、[Qmap 方法](tasks/QMAP_METHOD.md)。
- **规则与提示词：**[集中阅读副本](TASK_TEXTS.md)，原文在各任务 `prompts/`。
- **远端怎么连接、产物在哪里：**[开发机手册](DEVELOPMENT_MACHINES.md)。
- **协作方式：**[AGENTS.md](AGENTS.md)。

## 目录职责

| 目录 | 内容 |
|---|---|
| `tasks/` | 五任务入口、提示原文、小型示例、所选地图来源 |
| `experiments/` | 正式实现、配置、针对性检查和研究结果；按索引区分当前入口与历史依赖 |
| `grounded_llm/` | 已复用的基础组件与早期 harness；不是所有新任务的统一训练器 |
| `docs/` | 研究定义、唯一状态页、文献报告；`archive/` 与 `meetings/` 保留历史语境 |
| `runs/`、`logs/`、`outputs/`、`tmp/` | Git 外的运行产物与临时工作，不作为当前指令来源 |
| `data/`、`literature_review/` | Git 外的大型数据和文献工作材料 |

当前进度只更新状态页，数值比较见任务能力表和实验结果；本页不维护第二份计划。历史记录按需查询，不需要从头读完才能工作。

文献导航：[RELATED_WORK.md](RELATED_WORK.md)、[GCML 任务来源](docs/GCML_TASKS.md)。WorMI 有[详细精读](docs/WORMI_PAPER_READING_REPORT.md)和[分享稿](docs/WORMI_READING_REPORT.md)，用途不同，不是两份项目状态。
