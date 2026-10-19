# 按任务工作的统一入口

不同环境使用不同的规则外壳和状态编码；共同过程是：识别当前状态、目标与合法候选后继，读取后继到目标的地图距离关系，把选择对应到本轮动作。先分别训练、评测，再考虑多任务泛化与合并。

## 任务能力总表

指标为整题到达目标的成功率。空白表示尚无对应结果；计划任务尚未开展。
四个纯文字基线沿用原图：每任务40题，思考＋答案按16,000输出token计分。地图列填写各自已验证题集，分母随成绩标出，尚非统一40题横向比较；地图／LLM接图逐轮获得环境反馈，纯文字基线一次输出完整计划。

| 任务／目标范围 | DeepSeek | Qwen3-32B | 5.6 | 6.1 | 认知地图本身贪心 | 1.5B＋认知地图：单任务训 | 1.5B＋认知地图：统一训 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 积木：原八形状，12块清空 | 77.5% | 2.5% | 92.5% | 95% | **79.5%（159/200）** | | |
| 寻路：256节点 | 97.5% | 65% | 100% | 100% | **98.05%（251/256）** | | |
| 八盘汉诺塔：指定柱整塔终点 | 42.5% | 10% | 97.5% | 95% | **99.89%（19,659/19,680）** | | |
| 二阶魔方：复原 | 12.5% | 2.5% | 65% | 77.5% | **77.47%（2,846,397/3,674,160）** | | |
| 8-Puzzle：复原 | 22.5% | 15% | 82.5% | 90% | **98.28%（178,326/181,439）** | | |
| 积木：近清空非空目标，6–8步监测¹ | | | | | **90%（180/200）** | | |
| 积木：近清空非空目标，9–13步监测¹ | | | | | **68.5%（137/200）** | | |
| 积木：原地图SFT固定510题回归² | | | | | **82.55%（421/510）** | **94.12%（480/510，旧地图）** | |
| 过桥（计划） | | | | | | | |
| 推箱子（计划） | | | | | | | |
| 原五任务纯文字合计（200题） | **50.5%** | **19%** | **87.5%** | **91.5%** | | | |

¹ 积木地图统一采用用户指定的 **ffba71f结构化1000步候选**。近清空指目标还能用1–2次合法移除清空；本次目标保留原构造块，实际距离分别为7–8、9–11步，尚无6、12、13步成绩。两行是独立的新分布监测，不能代表任意近清空目标泛化，也不计入原五任务合计。
² 当前地图在原510题为421/510；1.5B单任务训480/510来自历史50k B2、旧地图986cb4d（该地图当时贪心397/510），到达且正确停止为475/510。不是当前地图重新SFT的结果，不构成单独的地图增益比较。其他空白不以旧任务、不同范围或中间训练指标补填。

采用权重与来源：[积木](blocks/Qmap/selection.json)、[寻路](path/Qmap/selection.json)、[汉诺塔](hanoi/Qmap/selection.json)、[二阶魔方](cube2/Qmap/selection.json)、[华容道](puzzle8/Qmap/selection.json)。寻路填留出测试；汉诺塔、魔方、华容道填全状态范围内复原，包含训练状态，非新状态泛化。
成绩依据：[纯文字四模型](../experiments/task_text_baselines/results/report.md)、[积木当前候选](../runs/qmap_dual_20261018/blocks12/branch_coverage_replay/run/evaluation/step_001000.json)、[积木近清空逐题](../runs/qmap_dual_20261018/blocks12/near_clear_readonly_pilot.json)、[历史1.5B单训](../experiments/map_sft_baselines/results/report.md)、[寻路](../experiments/path_qmap/results/summary.json)、[汉诺塔](../experiments/hanoi_qmap/results/full_coverage_summary.json)、[魔方repair_guard.final.policy](../runs/cube2_qmap/cycle_repair_f42c487/summary.json)、[华容道](../runs/qmap_dual_20261018/puzzle8/goal_expansion/stages/step_001000/metrics.json)。

后续原位填入同分布的新结果；统一训练指同一个1.5B读图模型跨任务训练，各任务Qmap独立。过桥、推箱子为现有待办占位，规则、难度和题集待确定。

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
