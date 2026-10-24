# 手电过桥

独立规则、数据适配和裁判已实现，首轮地图完成31/32题，尚未采用权重或准备正式SFT数据。已纳入下一阶段九类任务，使用正常人数与耗时，准备多目标Qmap、部分人员完整目标和新人数地图，见[训练设计](../QMAP_METHOD.md#任务扩展与训练安排)。现有[配置](../../experiments/new_tasks_qmap/configs/pilot/bridge.json)、[结果](../../experiments/new_tasks_qmap/results/report.md)和[历史文字筛查](../../experiments/task_text_baselines/results/report.md)分别保留，下一轮安排见[项目状态](../../docs/PROJECT_STATUS_AND_TODO.md)。

- [规则与输出](prompts/README.md)
- [数据](data/README.md)与[示例](data/example.json)
- [地图](Qmap/README.md)与[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/bridge/environment.py)

## 来源与适配

[Backhouse 与 Truong：The Capacity-C Torch Problem](https://people.cs.nott.ac.uk/psarb2/papers/abstract.html)。采用容量2、正整数时间的受限版本；本项目以完整状态图 Dijkstra 独立验算，不复刻论文中的优化算法。

## 变体与适用性

以下保留早期难度调查。

现有实现仍为两岸、一手电、容量2、全员到右岸。这个目标下存在成熟的最优调度结构；增加人数主要增加计划长度，长人数筛查中的完成失败、非最优成本与输出截断须分开。

| 候选 | 可核实来源 | 研究判断 |
|---|---|---|
| 容量C | [Backhouse与Truong，Science of Computer Programming2015](https://doi.org/10.1016/j.scico.2015.01.003)，提供动态规划与整数规划比较 | 有正式论文，但一般容量仍有平方时间算法；没有证据仅改容量就能产生适合本项目的困难分布 |
| 多岛、多手电、人员分赴不同目标 | [ČVUT PDDL规划课程原题](https://cw.fel.cvut.cz/wiki/courses/pui/assignments/assignment1-1) | 最贴近原任务的候选；人员、手电位置和图上调度共同决定动作价值。来源是公开教学任务，尚未核实到配套公开实现或成熟benchmark |
| 带安全／冲突约束的渡河 | [Csorba等：The Alcuin Number of a Graph](https://doi.org/10.1137/080736661)；[Rethinking the Illusion of Thinking论文及实现](https://github.com/11inaki11/Rethinking-The-Illusion-of-Thinking) | 前者支持一般冲突图及容量的理论分析，后者实现的是配对安全约束渡河；二者不是相同规则 |

ČVUT具体实例为6人、4岛、2手电，边为`i1–i2, i2–i3, i2–i4`，时间为1至6；人员最初都在i1，前三人目标i3、后三人目标i4。每人最多持一把手电，同岛可传递，同行至少一人持手电；仍每次最多两人、禁止并行。独立小型Dijkstra核对了课程三题的最优成本15、16、33；最后一题得到11次过桥加4次免费传递的见证，未证明该见证动作数最少。两手电视为可互换时，四岛实例的物理状态上界为`4^6 × C(6,2) = 61,440`，本地搜索展开57,231个状态，适合先核对精确真值。该核对仅验证规则可复现与规模，没有测试Flash或训练Qmap。

若采用多岛版本，地图输入需要包含人员所在岛、持灯人、桥连接与时间；目标需明确每人目的地及是否约束持灯人。只编码人员位置会丢失动作可行性。免费传递可使不同持灯状态的最短成本为零，所以成本关系一般是伪度量，零距离可对应多个持灯状态。目标未指定手电时仍是终态集合，当前将它保留为候选，规则、正式配置与采用权重均未切换。
