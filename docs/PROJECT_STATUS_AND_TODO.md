# 项目状态与有序 TODO

更新时间：2026-10-05；结果核对截止 03:00（北京时间）。本页是唯一当前进度页。研究定义见[研究简述](RESEARCH_BRIEF.md)，实际设置见[实验设计](../experiments/flamingo_map_reader/DESIGN.md)。

## 当前主线

主实验为单图寻路与 1000 初始棋盘积木的完整长轨迹 SFT。两任务都已完成 3 个 epoch，评测仍在进行；当前不修改训练／评测代码，不另启实验，监测由负责运行的 agent 继续处理。

| 设置 | 当前采用的范围 |
|---|---|
| 数据 | 寻路 4,192 个训练物理任务；积木 9,000 个，来自 1000 张初始棋盘 |
| 动作覆盖 | 训练候选覆盖寻路 768/768、积木 664/664；示范执行分别 767/768、664/664；不等于全部状态与目标组合已覆盖 |
| 训练 | 冻结 Qwen2.5-1.5B-Instruct 与地图，分别从头训练读取接口；固定编号语料 3 遍，实际 batch=1 |
| 监督 | 每轮排序与动作，到达后停止并总结；不输出数值距离；积木答案仅列最近 Top-10 加 current，输入保留全部合法候选 |

## 已核对结果及边界

| 证据 | 当前结果 | 能支持的判断 |
|---|---|---|
| 寻路 final reference | validation 6,140 个可解决策轮：动作可达率 100.0%，最短路动作率 74.4% | 同图未训练目标上的一步动作质量；随机合法动作可达率也为 100%，须结合最短路与闭环 |
| 寻路 final validation 闭环 | 全部 532 题：到达 423（79.5%），到达且最短 70（13.2%） | 已有完整长程证据；含 32 个初始即目标任务，非零步组另报 |
| 积木 epoch 2 reference | validation 6,409 个可解决策轮：动作可达率 85.5%，最短路动作率 77.7% | 随机动作可达基线为 50.3%；一步表现不能推算完整任务成功率 |
| 积木 final 闭环 | 首个 64 题分片：到达 3/64（4.7%），到达且最短 0/64 | 部分负结果保留；全部 1,100 题成绩留空 |

主结果只列 rollout 到达率／到达且最短路率，以及 reference 动作可达率／最短路动作率。任意同样最优的动作或路径都计对；reference 两率统一以作答前可解的决策轮为分母。排序、集合、停止和总结放入辅助表，动作与地图的一致性及消融另列地图使用证据。完整分母、结果和待填表统一见[结果报告](../experiments/flamingo_map_reader/results/readout_failure_analysis.md)。

## 文件、分支与运行定位

- 本轮训练协议为 `f73b700`，运行目录为 `/zhanghanyue/experiment/flamingo_map_reader/runs/long_f73b700_20261004`；合同、清单、源码职责见[实验 README](../experiments/flamingo_map_reader/README.md#分支源码与运行对应)。
- `codex/long-trajectory-training` 保存训练主线；`codex/full-trajectory-protocol` 已迁移到当前实现与文档。启动前旧方案用提交 `1ae0e0e` 识别，不再用分支名区分新旧。
- 本次从 35016 读取已有产物，确认寻路 final 为 72,576 步、积木 final 为 147,000 步；重启后的 40072 也已核对，同名已完成分片去重收录。
- 可达性字段在 `1c516e4` 引入，`07d21f9` 提供旧分片回填；正在运行的旧版评测仍可能产生缺字段分片。此次只读取记录与更新文档，没有执行回填。

## 有序 TODO

1. 由现有运行 agent 完成其评测队列；后续跨机器结果继续按分片键去重收录。
2. 先填积木 final reference 与完整 validation 闭环，再补两任务 epoch 1／2 闭环；积木分别报告训练棋盘新任务、新棋盘和初始即目标。
3. 收齐 final test、同编号 no_map 对照、重编号一致性与真实目标配对；合并案例后计算跨分片配对，不平均分片百分比。
4. 发布可达性成绩前核对字段覆盖，并同时报告全部决策轮、作答前可解子集与匹配随机基线。后续字段补齐由运行负责人处理。
5. 利用积木地图失败状态分析可行动作的地图排序，区分地图几何、并列选择与模型读取问题；该分析仍待进行，不在本次启动。
6. 据完整结果判断拟合、泛化和地图作用；本轮同时变动多个设置，不单独归因于数据量，也不由少量 checkpoint 判断是否饱和。

## 历史证据入口

历史结果保留在各自实验报告中；下表仅作索引，不重复维护数值。

| 历史研究 | 证据与用途 |
|---|---|
| 地图读取旧四图与 16／4 小样本 | [完整轨迹历史](../experiments/flamingo_map_reader/results/archive/full_trajectory_history.md)：局部拟合与早期失败诊断 |
| swap 及衍生分析 | [归档](../experiments/flamingo_map_reader/results/archive/swap_training.md)：不进入当前主证据链 |
| 图／积木连续 state token 对齐 | [图验收](../experiments/cml_map_scaling/results/step2_acceptance.md)、[积木读出](../experiments/state_interface_pilot/results/readout.md)：previous approach |
| 图 Q/V 转移与几何 | [Step 1 报告](../experiments/cml_map_scaling/results/report.md) |
| 旧单棋盘积木 Q/V | [试验结果](../experiments/external_map_interface/results/blocks_q_pilot.md)：转移拟合与目标距离分离 |
| 显式地图距离接口 | [五图报告](../experiments/external_map_interface/results/path256_five_graphs.md)、[早期单图](../experiments/external_map_interface/results/path256_distance.md)、[独立图复测](../experiments/external_map_interface/results/path256_diverse.md) |
| Qwen 4B／8B／32B 基线 | [验收报告](../experiments/qwen_path_blocks/results/report.md) |
| Sol、Luna、Flash 基线 | [Sol](../experiments/sol_dag_blocks/results/report.md)、[Luna](../experiments/gcml_counterexamples/results/summary.json)、[Flash](../experiments/gcml_counterexamples/results/deepseek_flash.json) |

[Related Work](../RELATED_WORK.md)维护相关文献，任务来源见 [GCML_TASKS.md](GCML_TASKS.md)。原工作区的自然目标 Q-only 草案未纳入本轮。
