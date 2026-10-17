# 项目状态与有序 TODO

本页只维护研究现状、决策和工作顺序。实验配方、指标表、恢复记录与运行日志由对应实验目录维护。

## 1. 当前研究问题

显式状态地图在充分任务监督之上，能否继续改善动作选择与连续完成？下一阶段进一步检验：多任务地图联合 SFT 能否改善已训练任务的 OOD 表现，并迁移到未参与任务 SFT、但提供地图的新任务。两层泛化都是待验证目标。定义与证伪标准见 [研究简述](RESEARCH_BRIEF.md)，2026-10-15 会议决策见[讨论稿](../experiments/map_sft_baselines/results/discussion_report.md#四2026-10-15-会议决策)。

## 2. 实验现状

| 研究 | 状态与定位 | 权威入口 |
|---|---|---|
| 同数据积木多基线 | 主实验与 7B R＋A 已完成；10k B2 诊断显示泛化差距，下一步区分数据与学习率时程影响 | [设计与执行](../experiments/map_sft_baselines/README.md)、[配置](../experiments/map_sft_baselines/configs/study.json)、[结果](../experiments/map_sft_baselines/results/report.md) |
| 冻结地图读取接口 | 既定训练与评测已完成，保留历史对照，不与新组直接混算 | [实验 README](../experiments/flamingo_map_reader/README.md)、[报告](../experiments/flamingo_map_reader/results/report.md) |
| 独立无地图任务 SFT | 历史强基线；已保存完整断点并退出，为新评测释放资源 | [协议](../experiments/llm_task_sft/README.md)、[总报告](../experiments/llm_task_sft/results/report.md) |
| 棋盘状态地图 | 提供冻结 Q 和独立地图贪心参照 | [协议与来源](../experiments/blocks_distance_map/README.md) |
| 汉诺塔／二阶魔方地图 | 汉诺塔六盘仍有循环。按用户决定，魔方采用循环修复对照的 `repair_guard` 最终断点作为当前复原模型；保持对称 Q 与 L∞。远距离复原和任意目标关系仍不充分，方向修正未通过循环验收。暂不长训 | [研究说明](../experiments/cube2_qmap/README.md)、[循环修复](../experiments/cube2_qmap/configs/cycle_repair.json)、[方向修正](../experiments/cube2_qmap/configs/cycle_projection.json) |
| 寻路纯 Q 地图 | 与积木、汉诺塔和魔方统一 MLP 骨干、距离与排序监督及 Q 槽位接口；节点独热输入，单独验收关系泛化和完整到达 | [实现与合同](../experiments/path_qmap/README.md) |

## 3. 已确定的比较原则

运行补充：三组新增训练（改进 B1、B2_10k、Qwen2.5-7B-Instruct R＋A）均已完成 20 epoch。028 训练后接续已实际启动并与 4090 共用评测池：改进 B1 最终完整到达 173/510，7B 已评点最高 470/510、最终 468/510；B2_10k 最终 RO 为 3/510，固定 64＋64 题的四检查点与输入干预均已完成：同更新数旧／新 B2 测试首步最短动作率为 96.88%／48.44%，新组的动作更受文字而非地图干预影响；502/510 题败于不可恢复动作。下一步应区分数据多样性与实际学习率时程，不再将少数重复样例解释为全局固定答案。详见[专项诊断](../experiments/map_sft_baselines/results/b2_10k_diagnosis.md)。已按授权删除原 reason-first B1／B2 checkpoint，其余 50k B0–B4（含改进 B1）保留整数 epoch 与已评最佳点，共清理约 319 GiB；核对时普通可用空间约 66 GiB。不以未评断点缺分代替零分。详见[当前实验报告](../experiments/map_sft_baselines/results/report.md)。

- 现有基线使用共同增强数据、相同任务输入与标准 assistant token 平均 CE；有地图输出排序、依据和动作，无地图输出依据和动作。会议提出的简化监督仍属后续候选，不覆盖既有运行。
- 地图版本统一且冻结；比较主干训练范围与接口条件。地图输入和排序监督一起变化，差值解释为整体方案收益。
- 监测题固定。现有平台期规则仍只作标记；常规收尾不另开会，异常情况再沟通。不因低 loss 或部分评测峰值宣称任务收敛。
- 历史实验与新实验分别记录。当前状态、动作后继由环境提供，现阶段不据此声称模型学会自主状态维护或内部搜索。

## 4. 有序 TODO

1. **收尾 B4 与改进 B1。**确认收敛后的表现；常规收尾无需专门会议，特殊情况再沟通。
2. **验证 10,000 条 B2。**已完成训练与首步／干预诊断，下一步应对齐学习率时程，区分数据多样性与调度影响；本次从既有数据抽样，后续数据不强制全部最短。
3. **补 7B 全参 SFT 基线。**已完成训练，使用 Qwen2.5-7B-Instruct 与相同 10,000 条物理轨迹，沿用简短 Reason＋动作；长 CoT 仍属后续候选。
4. **调研 FLAN。**确认数据与任务划分、实现和训练环境、迁移范围，以及泛化与遗忘的关系。
5. **扩充任务并推进联合训练。**魔方按全状态合同检查关系学习与完整到达，再决定是否接入联合 SFT；分训／合并权重未启动。其他任务按会议蓝图推进两层泛化。

## 5. 文档入口与历史证据

- [开发机手册](../DEVELOPMENT_MACHINES.md)：连接、目录与迁移。
- [相关工作](../RELATED_WORK.md)、[任务来源](GCML_TASKS.md)：文献与任务定义。
- 旧方案位于 `docs/archive/`，会议记录位于 `docs/meetings/`；均保留当时语境，不作为现行指令。
- 原两任务、融合 FFN 负结果与地图内容干预均由 [地图接口实验](../experiments/flamingo_map_reader/README.md)索引。
