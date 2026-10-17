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
| 汉诺塔／二阶魔方地图 | 现有纯 Q：汉诺塔关系轨道留出 108/108，魔方未见状态 138/512、长程仍弱。下一步分析魔方决策覆盖、重构广度／深度监督并重训，扩展汉诺塔至 4、5、6 盘；旧非纯 Q 实现及产物退出工作区 | [汉诺塔结果](../experiments/hanoi_qmap/results/report.md)、[魔方结果](../experiments/cube2_qmap/results/report.md)、[共享核心](../experiments/puzzle_qmap/README.md) |
| 寻路接口与早期方案 | 作为机制和任务边界的历史证据 | [实验目录](../experiments/README.md) |

## 3. 已确定的比较原则

运行补充：三组新增训练（改进 B1、B2_10k、Qwen2.5-7B-Instruct R＋A）均已完成 20 epoch。028 训练后接续已实际启动并与 4090 共用评测池：改进 B1 最终完整到达 173/510，7B 已评点最高 470/510、最终权重在评；B2_10k 最终 RO 仅 3/510，与低 loss 不一致，小样本确认平均 CE 掩盖首个排序数字错误：完整与截短前缀前向一致，关闭地图生成缓存也未改变输出；下一步应定位关键决策未学好的原因，不能直接归因于样本量。已按授权删除原 reason-first B1／B2 checkpoint，其余 50k B0–B4（含改进 B1）保留整数 epoch 与已评最佳点，共清理约 319 GiB；核对时普通可用空间约 66 GiB。不以未评断点缺分代替零分。详见[当前实验报告](../experiments/map_sft_baselines/results/report.md)。

- 现有基线使用共同增强数据、相同任务输入与标准 assistant token 平均 CE；有地图输出排序、依据和动作，无地图输出依据和动作。会议提出的简化监督仍属后续候选，不覆盖既有运行。
- 地图版本统一且冻结；比较主干训练范围与接口条件。地图输入和排序监督一起变化，差值解释为整体方案收益。
- 监测题固定。现有平台期规则仍只作标记；常规收尾不另开会，异常情况再沟通。不因低 loss 或部分评测峰值宣称任务收敛。
- 历史实验与新实验分别记录。当前状态、动作后继由环境提供，现阶段不据此声称模型学会自主状态维护或内部搜索。

## 4. 有序 TODO

1. **收尾 B4 与改进 B1。**确认收敛后的表现；常规收尾无需专门会议，特殊情况再沟通。
2. **验证 10,000 条 B2。**已完成训练，当前低评测读数须先审计一致性，再判断数据缩减的影响；本次从既有数据抽样，后续数据不强制全部最短。
3. **补 7B 全参 SFT 基线。**已完成训练，使用 Qwen2.5-7B-Instruct 与相同 10,000 条物理轨迹，沿用简短 Reason＋动作；长 CoT 仍属后续候选。
4. **调研 FLAN。**确认数据与任务划分、实现和训练环境、迁移范围，以及泛化与遗忘的关系。
5. **扩充任务并推进联合训练。**已有寻路、积木；按用户要求先完成三层汉诺塔和二阶魔方纯 Qmap，验证关系几何、对称性与真实合法后继上的连续到达，不添加转移预测头。其结果支持到什么程度，再决定接入联合 SFT；3×3 华容道类任务等仍为候选。按会议蓝图评测两层泛化。

## 5. 文档入口与历史证据

- [开发机手册](../DEVELOPMENT_MACHINES.md)：连接、目录与迁移。
- [相关工作](../RELATED_WORK.md)、[任务来源](GCML_TASKS.md)：文献与任务定义。
- 旧方案位于 `docs/archive/`，会议记录位于 `docs/meetings/`；均保留当时语境，不作为现行指令。
- 原两任务、融合 FFN 负结果与地图内容干预均由 [地图接口实验](../experiments/flamingo_map_reader/README.md)索引。
