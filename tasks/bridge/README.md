# 手电过桥

独立规则、数据适配和裁判已实现；首轮仅检验固定目标 Qmap 可行性。正式范围与预算见[配置](../../experiments/new_tasks_qmap/configs/bridge.json)，结果见[报告](../../experiments/new_tasks_qmap/results/report.md)。不作为多任务SFT已采用任务。

- [规则与输出](prompts/README.md)
- [数据](data/README.md)与[示例](data/example.json)
- [地图](Qmap/README.md)与[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/bridge/environment.py)

## 来源与适配

[Backhouse 与 Truong：The Capacity-C Torch Problem](https://people.cs.nott.ac.uk/psarb2/papers/abstract.html)。采用容量2、正整数时间的受限版本；本项目以完整状态图 Dijkstra 独立验算，不复刻论文中的优化算法。
