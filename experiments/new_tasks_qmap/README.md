# 五类新任务 Qmap 实验工作区

本目录维护推箱子、移车出库、棋子交换、过桥和21／24点的独立地图实验。**优化已暂停，推箱子搁置；其余地图按已验证范围保留。** 九类任务、小类及可用能力从[任务总览](../../tasks/README.md)阅读；本页只提供实验和实现入口。

## 当前结果入口

| 任务 | 当前采用情况 | 结果与限制 | 权重和正式配置 |
|---|---|---|---|
| 推箱子 | 固定墙体仍未通过，无采用地图 | [报告](results/sokoban/report.md) | [selection](../../tasks/sokoban/Qmap/selection.json)／[配置](configs/sokoban/) |
| 移车出库 | 原占格MLP限定选用；不覆盖可靠长程和新布局 | [报告](results/rush_hour/report.md) | [selection](../../tasks/rush_hour/Qmap/selection.json)／[配置](configs/rush_hour/) |
| 棋子交换 | 两棵10顶点树各自采用；最优效率未通过 | [报告](results/token_swapping/report.md) | [selection](../../tasks/token_swapping/Qmap/selection.json)／[配置](configs/token_swapping/) |
| 过桥 | 五人与六人分别采用，限定耗时分布 | [报告](results/bridge/report.md) | [selection](../../tasks/bridge/Qmap/selection.json)／[配置](configs/bridge/) |
| 21／24点 | 1–10任取四数的有限域、后缀和配置内中间目标采用 | [报告](results/game24/report.md) | [selection](../../tasks/game24/Qmap/selection.json)／[配置](configs/game24/) |

[五类结果摘要](results/report.md)只汇总当前结论。各分任务报告按“范围、关键成绩、限制、证据”组织；历史试验不再逐次写进正文。五类均尚无新增LLM读取验收。

## 文件职责

| 位置 | 保存什么 | 怎样使用 |
|---|---|---|
| `configs/<task>/` | 已提交的数据、采样、训练与验收合同 | 当前权重使用selection所指配置，其他配置用于历史复现 |
| `results/<task>/report.md` | 当前科学结论及改变判断的关键负结果 | 日常阅读入口 |
| `results/<task>/*.json` | 计数、误差、分层结果、比较条件与来源 | 由报告链接深入核对；不以文件名较新代替采用判断 |
| `results/shared/` | 跨任务标签、划分、接口及性能检查 | 性能优化与模型能力分开解释 |
| `configs/pilot/`、`results/pilot/` | 早期固定目标试验 | 历史固定目标复现 |
| `src/`、`tests/` | 实验编排及直接检查 | 复用任务规则与公共Qmap实现 |
| Git忽略的`runs/`、`tmp/`及远端运行目录 | 原始数据、曝光、日志、完整恢复断点、源码快照 | 保持原路径；机器连接与存储见[手册](../../DEVELOPMENT_MACHINES.md) |

整理只收束阅读入口，不移动已被配置、结果或checkpoint引用的文件。原始JSON、失败样本、旧配置及独有源码保留；采用记录是唯一模型身份入口。

## 运行与加载入口

当前不启动新训练。复现既有条件时，从仓库根目录使用原提交、配置及明确输出目录；先核对已有进程与产物，恢复必须使用包含优化器、采样及随机状态的完整`last.pt`。

```bash
python -m experiments.new_tasks_qmap.src.run --config CONFIG --out RUN_DIR --source-commit COMMIT --prepare-only
python -m experiments.new_tasks_qmap.src.run --config CONFIG --out RUN_DIR --source-commit COMMIT --device cuda:0 --resume
```

配置可能引用邻接运行的数据或初始化权重，复现时保留对应目录关系。只读推理从任务selection找到checkpoint，使用各任务的`MapReader`。

| 功能 | 实现入口 |
|---|---|
| 命令分流 | [run.py](src/run.py) |
| 完整状态对训练与数据 | [relations.py](src/relations.py)、[relation_data.py](src/relation_data.py) |
| 冻结评测与诊断 | [relation_evaluate.py](src/relation_evaluate.py)、[relation_diagnose.py](src/relation_diagnose.py) |
| 规则、特征及MapReader | [任务实现](../../grounded_llm/tasks/) |
| 编码器、距离、排序与采样 | [公共Qmap](../../grounded_llm/core/qmap/)、[方法说明](../../tasks/QMAP_METHOD.md) |

环境提供全部合法后继；单位成本任务比较预测剩余距离，过桥比较即时耗时加预测剩余成本。完整目标必须精确匹配。一般基本要求见[验收合同](configs/exact_pair_acceptance.json)，各任务额外的成本、效率或可达性条件以各自配置为准。
