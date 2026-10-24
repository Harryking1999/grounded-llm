# Grounded LLM with Cognitive Maps

研究显式状态地图能否改善 LLM 的状态关系判断、动作选择与任务泛化。环境提供真实状态和合法后继，各任务独立训练地图，同一个 LLM 通过 cross-attention 读取；研究边界见[研究简述](docs/RESEARCH_BRIEF.md)。

## 从这里进入

| 要了解什么 | 入口 |
|---|---|
| 主线、结论、接下来做什么 | [项目状态](docs/PROJECT_STATUS_AND_TODO.md) |
| 任务规则、数据、能力和所选地图 | [任务索引](tasks/README.md) |
| 代码在哪里、如何复用 | [实验与实现索引](experiments/README.md)、[Qmap方法](tasks/QMAP_METHOD.md) |
| 提示词原文 | 各任务 `prompts/README.md`；[TASK_TEXTS.md](TASK_TEXTS.md)为生成的集中副本 |
| 连接机器与找到运行产物 | [开发机手册](DEVELOPMENT_MACHINES.md) |
| Agent如何协作 | [AGENTS.md](AGENTS.md) |

## 目录职责

| 位置 | 内容 |
|---|---|
| `tasks/` | 十个任务的阅读入口；地图准备情况见任务索引 |
| `grounded_llm/core/` | 公共Q编码器、距离与采样、地图读取、SFT基础 |
| `grounded_llm/tasks/` | 任务规则、状态编码、标签、提示、数据适配与裁判 |
| `experiments/` | 实验配置、独立编排、结果与必要证据 |
| `docs/` | 研究定义、唯一状态页和文献报告；`archive/`与`meetings/`为历史材料 |
| `runs/`、`logs/`、`outputs/`、`tmp/` | Git外的运行产物和临时工作 |
| `data/`、`literature_review/` | 大型数据和文献工作材料；小型正式题集按原路径保留 |

## 代码依赖与接入

依赖方向为 `experiments → grounded_llm/tasks → core`；实验也可直接调用core。先找已有任务入口及直接依赖，再改对应层。Qmap训练按任务独立编排，共用数值组件；SFT共用编码、组批与训练。具体接口见[复用表](experiments/README.md#先复用这些实现)。

根目录旧harness和旧实验导入路径仍服务历史复现与缓存兼容，整理目录不迁移正在运行的源码、数据和checkpoint。

文献入口：[RELATED_WORK](RELATED_WORK.md)、[GCML任务来源](docs/GCML_TASKS.md)、WorMI[精读](docs/WORMI_PAPER_READING_REPORT.md)与[分享稿](docs/WORMI_READING_REPORT.md)。它们不维护当前计划。
