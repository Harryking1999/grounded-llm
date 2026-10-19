# 按任务工作的统一入口

不同环境使用不同的规则外壳和状态编码；共同过程是：识别当前状态、目标与合法候选后继，读取后继到目标的地图距离关系，把选择对应到本轮动作。先分别训练、评测，再考虑多任务泛化与合并。

## 任务能力总表

指标均为整题成功率，**每行明确题集与实际题数**。同一行未测试的模型用 `-` 占位；不同题集不横向相减。7B SFT为无地图基线；两列1.5B接图当前尚无正式对齐结果。

| 任务／目标范围／测试集 | DeepSeek | Qwen3-32B | 5.6 | 6.1 | 7B SFT | 认知地图本身贪心 | 1.5B＋认知地图：单任务训 | 1.5B＋认知地图：统一训 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **积木：12块清空，固定1000题（正式）** | - | - | - | - | - | **84.0%（840/1000）** | - | - |
| 积木：原文字40题，12块清空 | 77.5%（31/40） | 2.5%（1/40） | 92.5%（37/40） | 95%（38/40） | - | - | - | - |
| 寻路：原文字40题 | 97.5%（39/40） | 65%（26/40） | 100%（40/40） | 100%（40/40） | - | - | - | - |
| 寻路：固定留出256题 | - | - | - | - | - | **98.05%（251/256）** | - | - |
| 八盘汉诺塔：原文字40题 | 42.5%（17/40） | 10%（4/40） | 97.5%（39/40） | 95%（38/40） | - | - | - | - |
| 八盘汉诺塔：全状态到指定柱整塔，19680题 | - | - | - | - | - | **99.89%（19659/19680）** | - | - |
| 二阶魔方：原文字40题，复原 | 12.5%（5/40） | 2.5%（1/40） | 65%（26/40） | 77.5%（31/40） | - | - | - | - |
| 二阶魔方：全状态复原，3674160题 | - | - | - | - | - | **77.47%（2846397/3674160）** | - | - |
| 8-Puzzle：原文字40题，复原 | 22.5%（9/40） | 15%（6/40） | 82.5%（33/40） | 90%（36/40） | - | - | - | - |
| 8-Puzzle：全状态复原，181439题 | - | - | - | - | - | **98.28%（178326/181439）** | - | - |
| 过桥：计划，规则及题集待定 | - | - | - | - | - | - | - | - |
| 推箱子：计划，规则及题集待定 | - | - | - | - | - | - | - | - |

**积木历史回归（单独题集，非12块清空1000题）：**

| 测试集 | 当前地图贪心f9a1086 | 旧地图贪心986cb4d | 1.5B＋旧地图：单任务SFT | 7B无地图SFT |
|---|---:|---:|---:|---:|
| 原地图SFT固定510题 | 85.29%（435/510） | 77.84%（397/510） | 94.12%（480/510） | 92.16%（470/510） |

旧1.5B为50k B2，7B为10k R＋A最佳断点，训练条件不同；不能据分差单独归因地图。旧1.5B到达且正确停止为475/510。统一训指同一个1.5B读图模型跨任务训练，各任务Qmap仍独立。

积木正式范围仅清空，**正式完成须≥900/1000，当前未完成**；[固定1000题](blocks/data/clear1000.json)、[验收合同](../experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json)。当前840/1000；相对固定810参照丢89、新增119，净增30，历史510为435（丢28、新增39）。789基线及全部历史权重保留。当前图原文字40题尚未补测；旧ffba71f在该40题为27/40，不填作当前图成绩。尚无1.5B／7B在1000题上的正式结果；独立最终泛化仍待验收，见[模型比较](../experiments/blocks_distance_map/results/blocks12_joint_report.md)。

采用权重：[积木](blocks/Qmap/selection.json)、[寻路](path/Qmap/selection.json)、[汉诺塔](hanoi/Qmap/selection.json)、[二阶魔方](cube2/Qmap/selection.json)、[华容道](puzzle8/Qmap/selection.json)。全状态评测包含地图训练状态，不称独立新状态泛化；原文字每任务40题、思考＋答案16k token，其他任务验收范围保持原定义。

证据：[当前积木1000逐题](../runs/qmap_dual_20261018/blocks12/clear_forks_028_3000/evaluation/step_003000/results.json)、[纯文字四模型](../experiments/task_text_baselines/results/report.md)、[历史SFT](../experiments/map_sft_baselines/results/report.md)、[寻路](../experiments/path_qmap/results/summary.json)、[汉诺塔](../experiments/hanoi_qmap/results/full_coverage_summary.json)、[魔方](../runs/cube2_qmap/cycle_repair_f42c487/summary.json)、[华容道](../runs/qmap_dual_20261018/puzzle8/goal_expansion/stages/step_001000/metrics.json)。后续仅在相应题集行原位更新真实结果。

## 任务目录

| 任务 | 任务工作区 | 文字与提示 | 数据形态 | 地图 |
|---|---|---|---|---|
| 积木 | [blocks](blocks/README.md) | [prompts](blocks/prompts/README.md) | [data](blocks/data/README.md) | [Qmap](blocks/Qmap/README.md) |
| 寻路 | [path](path/README.md) | [prompts](path/prompts/README.md) | [data](path/data/README.md) | [Qmap](path/Qmap/README.md) |
| 汉诺塔 | [hanoi](hanoi/README.md) | [prompts](hanoi/prompts/README.md) | [data](hanoi/data/README.md) | [Qmap](hanoi/Qmap/README.md) |
| 二阶魔方 | [cube2](cube2/README.md) | [prompts](cube2/prompts/README.md) | [data](cube2/data/README.md) | [Qmap](cube2/Qmap/README.md) |
| 8-Puzzle（数字华容道） | [puzzle8](puzzle8/README.md) | [prompts](puzzle8/prompts/README.md) | [data](puzzle8/data/README.md) | [Qmap](puzzle8/Qmap/README.md) |

## 每个任务采用相同布局

```text
tasks/<task>/
  README.md             # 环境、已有代码、结果与正式配置的入口
  prompts/README.md     # 任务描述、规则、环境更新、各组输出与监督答案
  data/README.md        # 数据字段、准备代码与划分说明
  data/example.json     # 小型说明例；不是新增训练数据
  Qmap/README.md        # 共享方法、任务适配、训练与加载入口
  Qmap/selection.json   # 已采用权重的来源、适用范围和正式配置
  Qmap/checkpoints/     # Git 忽略：只放选定权重，不堆训练过程
```

根目录 [TASK_TEXTS.md](../TASK_TEXTS.md) 是所有任务文字的集中副本，由 `python -m tasks.export_texts` 从各任务的 `prompts/README.md` 与文字示例生成。修改任务内原文后重生成；不要直接编辑集中副本。`python -m tasks.export_texts --check` 检查是否同步。

## 文字分组

- 地图组／CoT 组：共享任务描述与规则，每轮收到真实环境更新。地图向量是地图组的独立输入，不把精确裁判距离放进文字。
- SOTA 组／32B 组：共享同一任务的描述与规则，一次输出完整剩余计划 JSON、无中途反馈。五任务文字已用于用户授权的纯文字基线，预算和题集见对应正式配置；不照搬旧整段计划结果充当匹配基线。
- 监督答案单列排序、Reason、Action。地图组是排序→R→A，无地图组删除排序行；当前为简短 Reason，长 CoT 尚未准备。

## 代码和权重怎么维护

`tasks/` 是日常任务入口；已有可复用实现与正式合同继续放在 `experiments/`。每个任务独立维护数据、训练与评测，基础 MLP、距离和排序组件共用，见 [Qmap 方法](QMAP_METHOD.md)。新训练仍写入 Git 外的 `runs/`；确认采用的模型才放入该任务的 `Qmap/checkpoints/`，来源写入 `selection.json`。整理入口不意味着清除旧产物或改变运行。

项目进展与下一步只维护在 [状态页](../docs/PROJECT_STATUS_AND_TODO.md)。历史结果仍按原合同解释。
