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
| `grounded_llm/core/` | Q 编码器、距离与损失、冻结读取、地图注意力、SFT 数据与训练基础 |
| `grounded_llm/tasks/` | 各任务规则、状态编码、标签与采样、提示、动作解析和裁判 |
| `experiments/` | 正式配置、任务独立的训练／评测编排和研究结果；旧导入路径保留兼容转发 |
| `grounded_llm/` 根目录旧模块 | 早期 harness；当前实现从 `core/` 和 `tasks/` 进入 |
| `docs/` | 研究定义、唯一状态页、文献报告；`archive/` 与 `meetings/` 保留历史语境 |
| `runs/`、`logs/`、`outputs/`、`tmp/` | Git 外的运行产物与临时工作，不作为当前指令来源 |
| `data/`、`literature_review/` | Git 外的大型数据和文献工作材料 |

当前进度只更新状态页，数值比较见任务能力表和实验结果；本页不维护第二份计划。历史记录按需查询，不需要从头读完才能工作。

## 代码依赖与接入

依赖方向是 `experiments → tasks → core`；实验也可以直接调用 core。公共核心不导入任务或实验，任务实现不导入实验。汉诺塔、魔方等 Qmap 训练仍独立编排，共享数值组件，不混成一个任务分支繁多的训练器。

| 工作 | 修改入口 |
|---|---|
| Q 编码器、距离和排序损失 | [core/qmap/model.py](grounded_llm/core/qmap/model.py) |
| 冻结推理；任务负责解释已有 checkpoint | [core/qmap/reader.py](grounded_llm/core/qmap/reader.py)、各任务 `model.py`／`reader.py` |
| 地图槽位、投影和 cross-attention | [core/reader](grounded_llm/core/reader) |
| 监督文本编码、数据组批、模型构造和 SFT 更新 | [core/sft](grounded_llm/core/sft)、[reader/batching.py](grounded_llm/core/reader/batching.py) |
| 规则、状态、采样、提示和裁判 | [tasks](grounded_llm/tasks) 中的对应任务 |
| 选择条件、预算、输出和运行方式 | 对应实验 `configs/` 和 `src/` 入口 |

任务提供 `Demonstration`，其中每轮包含文字、答案和地图 step；`encode_trajectory` 生成监督 token 与地图对应关系。`collate_examples` 将 `(encoded, steps)` 组批，公共 `train_update` 执行按监督 token 加权的更新。正式 SFT 使用 `train(config, baseline, model_path, data, out, pad_token_id=...)`；数据集提供这些样本及可更新的 `epoch`，任务需要的重编号等处理通过 `PreparedRoutes(transform=...)` 传入。现有积木命令已使用这条链路。

五任务地图接入同一 SFT 更新的 CPU 集成检查见 [test_shared_pipeline.py](tests/test_shared_pipeline.py)。这证明代码可以接通，不代表其他任务已有正式 SFT 数据、训练结果或跨任务坐标对齐。旧实验命令、checkpoint 参数键及轨迹缓存导入名保留；新代码直接使用上述入口，兼容文件不再添加实现。

文献导航：[RELATED_WORK.md](RELATED_WORK.md)、[GCML 任务来源](docs/GCML_TASKS.md)。WorMI 有[详细精读](docs/WORMI_PAPER_READING_REPORT.md)和[分享稿](docs/WORMI_READING_REPORT.md)，用途不同，不是两份项目状态。
