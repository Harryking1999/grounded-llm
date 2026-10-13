# 项目状态与有序 TODO

本页只维护研究现状、决策和工作顺序。实验配方、指标表、恢复记录与运行日志由对应实验目录维护。

## 1. 当前研究问题

显式状态地图在充分任务监督之上，能否继续改善动作选择与连续完成？需要分开判断地图是否有用、LLM 是否读取地图、以及读图是否带来匹配条件下的收益。定义与证伪标准见 [研究简述](RESEARCH_BRIEF.md)。

## 2. 实验现状

| 研究 | 状态与定位 | 权威入口 |
|---|---|---|
| 同数据积木多基线 | 当前主实验；共同数据已验收，首批四组已启动，后续组排队 | [设计与执行](../experiments/map_sft_baselines/README.md)、[配置](../experiments/map_sft_baselines/configs/study.json)、[结果](../experiments/map_sft_baselines/results/report.md) |
| 冻结地图读取接口 | 既定训练与评测已完成，保留历史对照，不与新组直接混算 | [实验 README](../experiments/flamingo_map_reader/README.md)、[报告](../experiments/flamingo_map_reader/results/report.md) |
| 独立无地图任务 SFT | 历史强基线；已保存完整断点并退出，为新评测释放资源 | [协议](../experiments/llm_task_sft/README.md)、[总报告](../experiments/llm_task_sft/results/report.md) |
| 棋盘状态地图 | 提供冻结 Q 和独立地图贪心参照 | [协议与来源](../experiments/blocks_distance_map/README.md) |
| 寻路接口与早期方案 | 作为机制和任务边界的历史证据 | [实验目录](../experiments/README.md) |

## 3. 已确定的比较原则

- 新基线使用共同增强数据、相同任务输入与标准 assistant token 平均 CE；有地图输出依据、排序和动作，无地图输出依据和动作。
- 地图版本统一且冻结；比较主干训练范围与接口条件。地图输入和排序监督一起变化，差值解释为整体方案收益。
- 监测题固定。平台期只作标记，停训由用户确认；不因低 loss 宣称任务收敛。
- 历史实验与新实验分别记录。当前状态、动作后继由环境提供，现阶段不据此声称模型学会自主状态维护或内部搜索。

## 4. 有序 TODO

1. **验收新基线的连续运行与评测链路。**具体进度、断点和首次完整评测更新到该实验报告，不追加到本页。
2. **比较统一数据后的收益。**分别检查主干解冻、接口容量、地图＋排序监督作用，以及共同状态判断与自主轨迹表现。
3. **按失败机制解释差异。**利用首次致命动作、停止错误、路线效率和地图内容探针，而非只比较 loss 或单步均值。
4. **用户决定是否停止平台期组。**保存实际训练长度与完整曲线；空闲资源接已登记的后续组。
5. **必要时补充种子与独立留出验证。**覆盖补充实验不默认启动；不事后换题直到地图获胜。

## 5. 文档入口与历史证据

- [开发机手册](../DEVELOPMENT_MACHINES.md)：连接、目录与迁移。
- [相关工作](../RELATED_WORK.md)、[任务来源](GCML_TASKS.md)：文献与任务定义。
- 旧方案位于 `docs/archive/`，会议记录位于 `docs/meetings/`；均保留当时语境，不作为现行指令。
- 原两任务、融合 FFN 负结果与地图内容干预均由 [地图接口实验](../experiments/flamingo_map_reader/README.md)索引。
