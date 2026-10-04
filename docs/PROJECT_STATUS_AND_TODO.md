# 项目状态与有序 TODO

更新时间：2026-10-05。本页是唯一当前进度页。研究问题见[研究简述](RESEARCH_BRIEF.md)，实际训练设置见[实验设计](../experiments/flamingo_map_reader/DESIGN.md)。

## 当前主线与已收录结论

**主实验已转为单图寻路与 1000 初始棋盘积木的完整长轨迹 SFT。**相比旧小样本，本轮扩大物理任务、沿途决策和编号覆盖，以完整逐轮及闭环评测回答“是否学会、是否泛化”。旧实验保留历史用途，不与本轮合并统计。

| 项目 | 当前采用的设置／证据 |
|---|---|
| 数据 | 寻路 4,192 个训练物理任务；积木 9,000 个，来自 1000 张初始棋盘 |
| 动作覆盖 | 训练候选覆盖寻路 768/768、积木 664/664；示范选择覆盖分别为 767/768、664/664 |
| 训练 | 冻结 Qwen2.5-1.5B-Instruct 与地图，两任务分别从头训练读取接口；固定编号语料训练 3 遍，实际 batch=1 |
| 监督 | 每轮排序与动作，到达后停止并总结；不输出数值距离；积木答案仅列最近 Top-10，输入仍为全部合法候选 |
| 已收录结果 | 寻路 reference：训练内 256 题与留出 532 目标，final 地图最优 78.7%／76.8%，闭环到达 202/256（78.9%）；积木 reference：训练内 256 题与留出 1,100 题，epoch 2 地图最优 40.5%／39.5% |
| 结论范围 | 寻路已用上地图，留出目标与训练内几乎重合；积木协议学会、动作远好于排序，但"内容接近随机"这一判断已收回——积木的排序类指标不是有效主证据，"动作后目标仍可解"尚未度量（[设计页三条口径](../experiments/flamingo_map_reader/DESIGN.md)）；积木的地图贪心上限按 group 拆开后是"训练棋盘 100%、各留出组 65.5–69.4%"，新棋盘与训练棋盘新任务几乎无差；两任务到最后一个 checkpoint 都在涨，未饱和；test、no_map 与编号／配对消融待补充 |

完整数据数量、划分、初始／更新文本、Top-10 原因及优化设置集中在[设计页](../experiments/flamingo_map_reader/DESIGN.md)。epoch 1 的同题对照和分母集中在[结果页](../experiments/flamingo_map_reader/results/readout_failure_analysis.md)，后续沿用同页补充。

## 运行与分支定位

- 当前实现分支：`codex/long-trajectory-training`；本轮训练协议提交：`f73b700`。
- 运行目录：`/zhanghanyue/experiment/flamingo_map_reader/runs/long_f73b700_20261004`。数据、合同、模型和结果文件的对应关系见[实验 README](../experiments/flamingo_map_reader/README.md#分支源码与运行对应)。
- 较早的 `1ae0e0e`（`codex/full-trajectory-protocol` 迁移前的尖端）是启动前方案，仍含已取消的数值距离监督；不能用该提交的文档判断当前运行是否启动或采用何种监督。
- `3992d6d`、`57d8ff1`、`d24e345`、`4ec7e78`、`aec7166`、`9f66c2b`、`f8161fb` 属于后续队列维护（重启、子进程认领、跨机器委派、每 epoch 验证闭环与排序）；截至本次整理，训练协议未随这些提交变化。旧 `long_9234c5c_20261003` 与原四图／小样本运行分别保留。

## 有序 TODO

1. 现有训练与监测继续由负责运行的 agent 推进；本次只整理文档，收录结果截止寻路 final、积木 epoch 2。
2. 将后续 epoch 的同题训练诊断及验证结果补入结果页，区分协议指标和关系／动作质量。
3. **补"动作后目标仍可解"的度量**，作为积木的主证据；现有排序类指标（`action_map_minimum`／`exact_ranking`／`current_relation`）降为诊断。补齐后再重估积木"内容接近随机"的判断。
4. **跑走死状态的排序探针**（纯地图计算，不用 LLM）：从地图贪心走死的积木留出状态出发，看地图给"仍可解"的走法排第几、离 argmin 差多少。用于分开"地图在积木上不可靠"与"argmin 不唯一、贪心采样运气差"两种解释。
5. 补齐最终留出任务、闭环、编号一致性、真实目标配对与无地图结果；积木按同板新任务／新棋盘及候选数量分层（评测已按 `group` 分层，取用即可）。
6. 据结果判断训练内拟合、泛化和地图作用；不把完整轨迹训练的收益单独归因于数据量，因为本轮监督格式等设置也有变化。

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
