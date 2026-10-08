# 项目状态与有序 TODO

更新：2026-10-08。本页记录项目动机、整体进展和研究优先级；当前实验设置见 [DESIGN](../experiments/flamingo_map_reader/DESIGN.md)，新运行结果见[本轮报告](../experiments/flamingo_map_reader/results/report.md)。上一轮[设计](../experiments/flamingo_map_reader/DESIGN_long_trajectory.md)与[报告](../experiments/flamingo_map_reader/results/report_long_trajectory.md)单独保留。

## 1. 为什么做这个项目

LLM 已有较强的语言理解和推理能力，但在复杂任务中，仍可能难以稳定地认识当前状态、判断动作后果，并完成连续决策。前期实验既发现了状态报告错误，也发现了“状态报告正确，却选择非法动作或走入死局”的情况。这提示我们：知道任务的语言描述、表示任务状态和利用状态作出决策，是需要分别检验的能力。

项目的出发点是：**能否给已有语言智能补上一种可用的状态认知，使模型原有的推理能力更好地作用于任务世界？** 受认知地图启发，我们希望外部模块不仅表示“现在是什么状态”，还提供“动作会带来什么后果”以及“哪些方向更接近目标”的信息，让 LLM 理解任务、读取这些关系并选择行动。

长期希望形成“观察与更新状态 → 推演动作后果 → 选择并执行 → 用反馈校准”的循环。近期先回答更具体的问题：小型 LLM 能否通过可学习接口有效使用已有地图，以有限的适配训练改善复杂任务表现，并在匹配条件下与更大模型、直接任务微调及强推理模型比较。这是研究目标，尚不是已完成的结论。

上述动机与路线来自飞书[《Grounded LLM方案》](https://zcnhpsd26tyi.feishu.cn/wiki/X9NZw0K4zizI2vkM14OcK4tXnXd)、[《蓝图》](https://zcnhpsd26tyi.feishu.cn/wiki/ZiGywBK7Iiej98k7JnRcKsounec)和[《LLM读取认知地图-方案》](https://zcnhpsd26tyi.feishu.cn/wiki/ZNNSwpIo0iHwMgk7mk7coBA3nzb)。

## 2. 当前切入点与任务分工

当前主线是 **LLM＋外部认知地图**：地图提供状态与关系，可训练接口负责接入，LLM 负责结合任务要求生成语言和动作。先把这条信息流打通，再讨论更通用的状态建模与规划。

两个任务分别强调不同问题：

- **积木：复杂状态与动作后果。** 每次移除都会改变剩余几何结构和后续可行性，用它检查模型能否利用状态信息，在多个合法动作中避免破坏目标。
- **寻路：状态关系与目标方向。** 当前节点容易描述，难点是判断下一步如何接近目标，用它检查地图的远近关系能否转化为有效决策。

本阶段冻结 LLM 和地图，通过 cross-attention 训练读取接口。环境提供当前状态、合法候选与执行反馈，因此现阶段主要检验地图读取和使用；自主感知、学习状态更新及内部搜索仍是后续问题。地图＋贪心提供同任务上的参照，重点是弄清地图能力接入 LLM 后保留了多少、差距来自哪里。

早期方案曾设想只训练状态报告，再检验能否迁移到动作选择；当前完整轨迹 SFT 已直接包含动作监督。因此，当前成绩回答的是“接口训练后能否读取并利用地图行动”，尚未验证“仅学会读状态就能迁移到未训练的决策能力”。这个更强的泛化设想仍保留为后续研究问题。

## 3. 整体进展：从失败现象到地图使用

| 阶段 | 已有进展 | 仍需回答的问题 |
|---|---|---|
| 识别语言模型的困难 | 建立寻路、积木基线，观察到状态错误、非法动作、死局与提前停止 | 哪些失败来自状态信息不足，哪些来自信息已有却未被正确使用？ |
| 建立有用的地图 | 完成图 Q/V 与积木共享编码器实验；积木引入距离与可达性监督，已有跨初始棋盘的地图证据 | 转移拟合、目标关系与行动方向并不等价；地图在何种任务和状态上可靠？ |
| 让 LLM 读取并行动（当前重点） | 已实现 cross-attention 接口和完整轨迹训练；寻路闭环表现明显改善，积木参考状态下的动作选择改善，但自主连续执行仍弱 | 模型是否实际依赖地图？地图误差、读取误差与自身轨迹上的错误分别贡献多少？ |
| 验证方法优势与泛化 | 已有留出任务评测，上一轮地图倒序与早期安全前缀干预已完成；no_map 已停止 | 相对无地图训练、文字状态接口及更强模型，收益是否成立？能够迁移到哪些新条件？ |

上一轮寻路与积木已完成既定训练，积木 final 权重已发布，已有评测结果见上一轮报告。本次积木按会议决议重新训练与评测，设置见下节。寻路目前只训练一张地图；积木新棋盘上的表现也要区分地图训练与接口训练的曝光范围。已有进展不能直接扩展为跨图、跨动作集合或跨任务泛化。

旧连续 token 接口、小样本和失败变体保留为历史证据。它们用于解释路线如何演进，不与当前协议合并统计；各实验入口见[实验目录](../experiments/README.md)。

## 4. 有序 TODO

**当前决议：停止融合 FFN 训练与旧评测队列，恢复原分离 K/V、无 FFN 接口，从头训练。** 依据 2026-10-08《会议.md》及用户补充：普通成功任务保留原编号增强；已达标与无解两类不增强、数量保持一致。复用原缓存、物理任务、提示和标签，不加载任何旧训练权重。

正式训练与评测分别见 [blocks_kv_restart.json](../experiments/flamingo_map_reader/configs/blocks_kv_restart.json) 和 [blocks_kv_evaluation.json](../experiments/flamingo_map_reader/configs/blocks_kv_evaluation.json)。原代码仍由 `AddressedMapMemoryEncoder` 和统一 `build_reader` 选择，分离路径始于 `1971619`，无需复制旧源码。

旧运行 `blocks_ffn_failure_batch4_75cc4fd` 已停止，停机时约 5.36 epoch；原运行目录、权重、原始回答、缓存与日志完整保留并作归档标记。已全量评齐的 0.5–5 epoch [报告](../experiments/flamingo_map_reader/results/report_ffn_failure.md)、[计数摘要](../experiments/flamingo_map_reader/results/blocks_ffn_failure_summary.json)和[历史设计](../experiments/flamingo_map_reader/DESIGN_ffn_failure.md)供参考。epoch 5 普通完成 4/510，初始目标 99/100，固定失败无解 97/100；旧三任务加权与旧 NDCG 只保留在该归档中。不能由这些结果断定 FFN 是失败原因。

1. **训练与四卡评测正常推进。** 2026-10-08 22:40（北京时间，远端日志时间），训练已至 step 39,503（3.16024 epoch），最近 500 步平均 loss 0.2419、未见非有限值；最新完整断点为 38,750。此前从 32,500 恢复的 1,101 步已全部重算。完整评测分片增至 55：0.5 epoch R/F 各 320/510，1 epoch 各 192，1.5 epoch R 192/F 128，2 epoch R 510/F 384，2.5 epoch R/F 均为 510，且两个独立终止任务各完成 100 题。2.5 epoch rollout 完成 23/510，单步可达 61.5% 对同状态随机 47.4%，位置倒数 NDCG@10 为 0.547 对 0.479，距离倒数 @10 为 0.864 对 0.860。详见[当前报告](../experiments/flamingo_map_reader/results/report.md)，各断点覆盖不同，不能当作全量学习曲线。
2. **继续收齐各断点三类任务。** 2.5 epoch 已全量评齐普通任务、独立初始目标与固定失败上下文；已核对同期 FFN 的编号、完整提示历史与终止状态一致。初始目标正确停止 K/V 93/100、FFN 100/100，终止摘要正确均为 93/100；固定无解判断 92/100 对 97/100。两类是固定上下文的 reference 判断，不填作生成整局成功率，不与普通路径任务混算。当前四卡生成 3 epoch 的前两片 R/F，尚无完整分片；其余断点按既定队列收齐。
3. **保留局部改善与反向结果。** 固定 128 题的 K/V 完成仍为 2 epoch 的 2/128→2.5 epoch 的 9/128；当前全量同题 510 题完成为 K/V 23/510、FFN 7/510，首轮可达 54.5%／51.2%（随机 47.9%）、地图最优 5.9%／2.2%（随机 1.81%）。255 个新棋盘任务完成为 12/255 对 4/255，同棋盘为 11/255 对 3/255。完整 reference 510 题的全轮可达反而 FFN 更好（80.6% 对 K/V 65.1%），普通已达标正确停止为 K/V 342/345 对 FFN 324/345；相同 reference 510 题中，2→2.5 epoch 的 FFN 可达由 63.9%→80.6%，K/V 由 63.0%→65.1%。K/V 首轮 1–3／1–10 为 66.1%／87.3%，FFN 有 37.3% 选编号 4；各自 rollout 死局无解报告为 43/344 对 71/398，远低于固定无解测试，不能相互替代。保留格式失败、缺失随机字段的环境补算与所有计数；新旧训练权重、预算及架构共同变化，不作纯架构因果结论。详见[最新同题比较](../experiments/flamingo_map_reader/results/report.md)。
4. **持续跟进三代对照。** 已按用户要求设置当前线程每 30 分钟跟进，仅在有新增完整评测、关键变化或异常时汇报。比较当前 K/V 重训、归档 FFN，以及最初 `long_f73b700_20261004/blocks`。同题 1 epoch 首轮 rollout，当前 K/V 保持可达 105/192、FFN 113/192；选择编号 1–3 为 69.3%/70.3%，随机期望仅 5.4%。最初 K/V 的首轮可达由 epoch 1 的 58.1% 至 epoch 5 的 70.3%，选择 1–3 从 71.0% 至 23.2%；该运行 batch 1、每 epoch 49,000 步、旧 validation 与当前 test 无物理任务交集，按相近更新数和训练阶段分别作历史参照，不作纯架构因果结论。详细基线与边界见[当前报告](../experiments/flamingo_map_reader/results/report.md)。优先复用旧回答与 CPU 重评分；同题跨权重新推理、无地图、文字状态接口、更强模型和干预另定范围，不自动恢复 no_map，不增加未经授权的研究分叉。

## 5. 长期研究空间与文档入口

飞书蓝图还提出三条相互关联的方向：分析 LLM 内部是否已有支持决策的状态关系结构；研究怎样训练兼具状态、转移与目标方向信息的地图；利用语言知识构建更可泛化的认知地图。当前读取实验为这些问题积累依据，后续是否展开以及先做哪条，由结果决定。

- [研究简述](RESEARCH_BRIEF.md)：当前假设、信息条件和证据边界。
- [当前实验设计](../experiments/flamingo_map_reader/DESIGN.md)与[本轮报告](../experiments/flamingo_map_reader/results/report.md)：当前分离 K/V 重训协议和结果；[上一轮报告](../experiments/flamingo_map_reader/results/report_long_trajectory.md)记录原两任务成绩。
- [实验 README](../experiments/flamingo_map_reader/README.md)：代码、运行和汇总入口；[积木地图说明](BLOCKS_QMAP_PLAN.md)：地图来源。
- [相关工作](../RELATED_WORK.md)与[任务来源](GCML_TASKS.md)：文献和任务依据。

### 历史证据入口

[早期动机与基线汇报](archive/early_experiment_summary.md)、[旧接口计划](archive/EXPERIMENT_TODO.md)、[完整轨迹旧实验](../experiments/flamingo_map_reader/results/archive/full_trajectory_history.md)、[失败变体](../experiments/flamingo_map_reader/results/archive/abandoned_readout_variants.md)。归档保留当时语境，不维护当前进度。
