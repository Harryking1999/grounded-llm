# 手电过桥

**五人和六人分别有一张已采用Qmap**，均通过配置内新耗时组合的独立完成与成本验收。规则是两岸、一手电、每次1或2人，耗时为同行者最大值；完整目标指定每个人及手电的位置。全员到右岸、部分人员分布目标和中间起点均已检查。

同人数地图覆盖已配置的互异正整数耗时，人员按耗时升序编号；重复耗时、人物重排、范围外耗时和其他人数尚无采用结论。两张地图的具体结果见[报告](../../experiments/new_tasks_qmap/results/bridge/report.md)，权重与配置见[selection](Qmap/selection.json)。新增LLM读取尚未验收，优化已暂停。

- [规则与输出](prompts/README.md)
- [数据与划分](data/README.md)、[示例](data/example.json)
- [地图编码与读取](Qmap/README.md)、[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/bridge/environment.py)

## 来源与适配

[Backhouse与Truong：The Capacity-C Torch Problem](https://people.cs.nott.ac.uk/psarb2/papers/abstract.html)。采用容量2的受限版本，以完整带权状态图Dijkstra独立验算，不复刻论文的优化算法。容量变化、多岛、多手电或带安全约束渡河不属于当前采用任务；历史调查不作为已实现能力。
