# 项目状态与有序 TODO

本页只维护研究现状、决策和工作顺序。实验配方、指标表、恢复记录与运行日志由对应实验目录维护。

## 1. 当前研究问题

显式状态地图在充分任务监督之上，能否继续改善动作选择与连续完成？下一阶段进一步检验：多任务地图联合 SFT 能否改善已训练任务的 OOD 表现，并迁移到未参与任务 SFT、但提供地图的新任务。两层泛化都是待验证目标。定义与证伪标准见 [研究简述](RESEARCH_BRIEF.md)，2026-10-15 会议决策见[讨论稿](../experiments/map_sft_baselines/results/discussion_report.md#四2026-10-15-会议决策)。

## 2. 实验现状

| 研究 | 状态与定位 | 权威入口 |
|---|---|---|
| 同数据积木多基线 | 当前主实验进入收尾；优先确认 B4、改进 B1 的完整结果，再验证小数据 B2 与强 CoT 基线 | [设计与执行](../experiments/map_sft_baselines/README.md)、[配置](../experiments/map_sft_baselines/configs/study.json)、[结果](../experiments/map_sft_baselines/results/report.md) |
| 冻结地图读取接口 | 既定训练与评测已完成，保留历史对照，不与新组直接混算 | [实验 README](../experiments/flamingo_map_reader/README.md)、[报告](../experiments/flamingo_map_reader/results/report.md) |
| 独立无地图任务 SFT | 历史强基线；已保存完整断点并退出，为新评测释放资源 | [协议](../experiments/llm_task_sft/README.md)、[总报告](../experiments/llm_task_sft/results/report.md) |
| 棋盘状态地图 | 提供冻结 Q 和独立地图贪心参照 | [协议与来源](../experiments/blocks_distance_map/README.md) |
| 汉诺塔／二阶魔方地图 | 四／五／六盘汉诺塔留出关系到达为 960/960、1020/1024、985/1024，六盘经典题仍循环。六动作局部图及在线共同左乘已训练完成：同题到达新队列 131 对 120/512，旧队列 151 对 151/512；中层单步最优率仍弱，381 个失败全部往返循环。用户已要求针对性修复，当前对比绝对距离与动作距离差 Huber，保持纯 Q 及完整软排序；旧非纯 Q 实现及产物已退出工作区 | [进展监督合同](../experiments/cube2_qmap/configs/quarter_turn_progress.json)、[汉诺塔结果](../experiments/hanoi_qmap/results/report.md)、[魔方结果与审计](../experiments/cube2_qmap/results/report.md) |
| 寻路接口与早期方案 | 作为机制和任务边界的历史证据 | [实验目录](../experiments/README.md) |

## 3. 已确定的比较原则

- 现有基线使用共同增强数据、相同任务输入与标准 assistant token 平均 CE；有地图输出排序、依据和动作，无地图输出依据和动作。会议提出的简化监督仍属后续候选，不覆盖既有运行。
- 地图版本统一且冻结；比较主干训练范围与接口条件。地图输入和排序监督一起变化，差值解释为整体方案收益。
- 监测题固定。现有平台期规则仍只作标记；常规收尾不另开会，异常情况再沟通。不因低 loss 或部分评测峰值宣称任务收敛。
- 历史实验与新实验分别记录。当前状态、动作后继由环境提供，现阶段不据此声称模型学会自主状态维护或内部搜索。

## 4. 有序 TODO

1. **收尾 B4 与改进 B1。**确认收敛后的表现；常规收尾无需专门会议，特殊情况再沟通。
2. **验证 10,000 条 B2。**检查数据缩减后到达率能否仍达九十多；后续数据不强制全部最短。
3. **补 7B CoT 全参 SFT 基线。**会议规格见讨论稿；后续考虑地图组“排序＋A”、无地图组“CoT＋A”。
4. **调研 FLAN。**确认数据与任务划分、实现和训练环境、迁移范围，以及泛化与遗忘的关系。
5. **扩充任务并推进联合训练。**已有寻路、积木；汉诺塔三至六盘及二阶魔方纯 Qmap 已运行。先依据冻结报告中的六盘循环和魔方欠拟合，决定是否及如何接入联合 SFT；不能将补齐数据覆盖视为可靠求解，也不添加转移预测头。3×3 华容道类任务等仍为候选。按会议蓝图评测两层泛化。

## 5. 文档入口与历史证据

- [开发机手册](../DEVELOPMENT_MACHINES.md)：连接、目录与迁移。
- [相关工作](../RELATED_WORK.md)、[任务来源](GCML_TASKS.md)：文献与任务定义。
- 旧方案位于 `docs/archive/`，会议记录位于 `docs/meetings/`；均保留当时语境，不作为现行指令。
- 原两任务、融合 FFN 负结果与地图内容干预均由 [地图接口实验](../experiments/flamingo_map_reader/README.md)索引。
