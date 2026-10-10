# 项目状态与有序 TODO

更新：2026-10-10。本页记录项目动机、整体进展和研究优先级；当前实验设置见 [DESIGN](../experiments/flamingo_map_reader/DESIGN.md)，新运行结果见[本轮报告](../experiments/flamingo_map_reader/results/report.md)。上一轮[设计](../experiments/flamingo_map_reader/DESIGN_long_trajectory.md)与[报告](../experiments/flamingo_map_reader/results/report_long_trajectory.md)单独保留。

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

**当前决议：停止融合 FFN 训练与旧评测队列，恢复原分离 K/V、无 FFN 接口，从头训练。** 依据 [2026-10-08 会议记录](meetings/2026-10-08.md)及用户补充：普通成功任务保留原编号增强；已达标与无解两类不增强、数量保持一致。复用原缓存、物理任务、提示和标签，初始重训不加载任何旧训练权重；本轮已完成 10 epoch，用户随后授权从本轮完整断点继续到 20 epoch。

正式训练与评测分别见 [blocks_kv_restart.json](../experiments/flamingo_map_reader/configs/blocks_kv_restart.json) 和 [blocks_kv_evaluation.json](../experiments/flamingo_map_reader/configs/blocks_kv_evaluation.json)。20 epoch 扩展与两节点分工见 [续训预算](../experiments/flamingo_map_reader/configs/blocks_kv_twenty_epoch_extension.json)及[两节点合同](../experiments/flamingo_map_reader/configs/blocks_kv_two_node_continuation.json)。原代码仍由 `AddressedMapMemoryEncoder` 和统一 `build_reader` 选择，分离路径始于 `1971619`，无需复制旧源码。

旧运行 `blocks_ffn_failure_batch4_75cc4fd` 已停止，停机时约 5.36 epoch；原运行目录、权重、原始回答、缓存与日志完整保留并作归档标记。已全量评齐的 0.5–5 epoch [报告](../experiments/flamingo_map_reader/results/report_ffn_failure.md)、[计数摘要](../experiments/flamingo_map_reader/results/blocks_ffn_failure_summary.json)和[历史设计](../experiments/flamingo_map_reader/DESIGN_ffn_failure.md)供参考。epoch 5 普通完成 4/510，初始目标 99/100，固定失败无解 97/100；旧三任务加权与旧 NDCG 只保留在该归档中。不能由这些结果断定 FFN 是失败原因。

1. **报告固定为一张 0.5–20 epoch 总表。** 按用户 2026-10-09 指令，[本轮报告](../experiments/flamingo_map_reader/results/report.md)只维护顶部总表与一行当前状态，不再逐次追加汇报。列出 rollout 完整路径到达率；普通 8000 类的单步可达率、地图最优率、位置倒数与 Q-map 距离归一化 NDCG@10 及各自 @1；两个 1000 类的到达判断与无解判断。距离归一化替代原距离倒数，正式定义见评测合同和 DESIGN；已有完整分片已通过保存回答重评分。R/F 与评测题数明确标注，缺失填 —。首轮、编号、随机与反序基线、FFN 与最初 K/V 对照留在现有 JSON 摘要，需要时单独查询；旧距离倒数只作历史兼容，不混用公式。
2. **41511 四卡续训至 20 epoch，35016 两卡与 4090 单卡协同评测。** 2026-10-10 两节点重启后，按用户指定将四卡节点 41511 用于训练：四张 A800 从最新完整 checkpoint-236250（18.9 epoch）恢复 optimizer、scheduler、trainer state 和四份 RNG，每卡 batch=1、全局 batch=4，原 warmup 6250、目标250000步均不变；停机前236281步，重跑31步。已验收恢复后236287→236378步增长（18.910 epoch）。35016现有两张A100主评测，一路rollout、一路reference，模3余数0/1；4090第4号卡余数2保持不变。文件同步已在35016恢复，primary/auxiliary均running，已回传完整辅助分片。原完整结果复用，中断分片保留后重做。以下指标与ETA保留为此前11:49验收快照，不代表重启后的新统计；减少主评测卡数后，旧整表ETA需要后续重新估算。表内汇总 453/800 个完整分片（56.6%）。14.5 epoch 普通 R/F 为 510/510 题，到达 161/510（31.6%），完成并正确停止 155/510（30.4%）；同题 10 epoch 到达 109/510，救回 88、退步 36，净增 52。16 epoch 普通 R/F 为 320/384 题，到达 125/384（32.6%），完成并正确停止 102/384（26.6%）；同题 10 epoch 到达 87/384，救回 74、退步 36，净增 38。16.5 epoch 普通 R/F 为 448/448 题，到达 152/448（33.9%），完成并正确停止 148/448（33.0%）；同题 10 epoch 到达 99/448，救回 86、退步 33，净增 53。17 epoch 普通 R/F 为 256/320 题，到达 100/320（31.2%），完成并正确停止 96/320（30.0%）；同题 10 epoch 到达 66/320，救回 62、退步 28，净增 34。14.5 epoch仍是最新普通R/F均完整的510题结果；15–17 epoch尚未评齐，不能直接比较不同题集均值。16.5 epoch与同448题的14.5 epoch到达147→152，救回62、退步57、净增5；17 epoch与同320题的14.5 epoch到达90→100，救回47、退步37、净增10。17.5 epoch已发布并进入队列，但本次尚无完整分片；13.5 epoch仍无完整分片。11–17.5 epoch独立起始已达标与固定无解类别尚无完整分片，不沿用10 epoch终止成绩。自身执行到达后正确停止：16 epoch为102/125（81.6%），16.5 epoch为148/152（97.4%），17 epoch为96/100（96.0%）；各自题集不同。17 epoch未达标宣告成功170次，其中168次已不可达、2次仍可达；成功停止中真正达标96/266（36.1%）。无解声明中真正无解28/31（90.3%），3次已达标却声明无解；已无合法动作且未达标时正确声明无解28/154（18.2%）。16 epoch之前观察到的23次达标误判无解仍保留。训练实测约 0.70 秒/更新，剩余约 5.8 小时，20 epoch 训练粗估 10 月 10 日 16–20 时完成，最新权重三类评测随后约 2–4 小时。整表纯生成工作量约 20.6 小时，含载入、同步与任务差异粗估还需 25–32 小时，约 10-11 12:49 至 10-11 19:49，节点持续可用时；节点到期与资源排队会继续延后。沿用模 3 互斥分片、每半 epoch、最新优先；完整辅助输出每 10 分钟回传。到期恢复自动化当前为暂停状态，本次保持暂停并更新节点分工；未恢复固定间隔指标汇报。 训练此前到期重启从 checkpoint-190000 恢复 optimizer、scheduler、trainer state 和四卡 RNG，原 warmup 6250 步；重启前未保存的 913 步重跑，原 10 epoch final 和完整权重保留。运行源码 fb923f8，文件同步使用可复用 artifact_bridge；不新增节点专属启动源码。只计完整分片，部分结果保留真实分母。

3. **保留比较边界。** FFN 同题同期对照最新至 5 epoch；最初 K/V 的 validation 与本轮 test 无物理交集，batch、数据与预算不同。旧 NDCG 未按新公式重评时保持缺失，不拿总体 reference 当首轮，不由不匹配均值推断架构因果收益。2026-10-09 用户授权的纯地图同题 CPU 对照完成：现用树关系 Q 到达 345/510（67.6%），补清空与长程关系的续训 15,000 步 Q 到达 397/510（77.8%）；新救回 105 题、退步 53 题，8 步改善最大，6 步略退步。新权重仍未接入 LLM；原地图生成的 10,000 条物理训练任务与缓存监督继续保留。已保存并合入地图分支的评测工具，权重差异见[地图文档末尾](BLOCKS_QMAP_PLAN.md#2026-10-09-补充更好的续训权重尚未接入-llm)。新地图补充监督与这些任务的重合尚未审计，不能把这次对照当作纯跨棋盘泛化证明。合同见 `blocks_map_baseline_comparison.json`，汇总与分组结果在现有 JSON 的 `map_greedy_comparison`。原始产物与此前冻结计数保留在摘要及 Git 历史。

## 5. 长期研究空间与文档入口

飞书蓝图还提出三条相互关联的方向：分析 LLM 内部是否已有支持决策的状态关系结构；研究怎样训练兼具状态、转移与目标方向信息的地图；利用语言知识构建更可泛化的认知地图。当前读取实验为这些问题积累依据，后续是否展开以及先做哪条，由结果决定。

- [研究简述](RESEARCH_BRIEF.md)：当前假设、信息条件和证据边界。
- [当前实验设计](../experiments/flamingo_map_reader/DESIGN.md)与[本轮报告](../experiments/flamingo_map_reader/results/report.md)：当前分离 K/V 重训协议和结果；[上一轮报告](../experiments/flamingo_map_reader/results/report_long_trajectory.md)记录原两任务成绩。
- [实验 README](../experiments/flamingo_map_reader/README.md)：代码、运行和汇总入口；[积木地图说明](BLOCKS_QMAP_PLAN.md)：地图来源；[终止判断诊断](../experiments/flamingo_map_reader/results/termination_diagnostic.md)：固定终止题与自身执行的差别、指标分母及真实失败例子。
- [相关工作](../RELATED_WORK.md)与[任务来源](GCML_TASKS.md)：文献和任务依据。

### 历史证据入口

[早期动机与基线汇报](archive/early_experiment_summary.md)、[旧接口计划](archive/EXPERIMENT_TODO.md)、[完整轨迹旧实验](../experiments/flamingo_map_reader/results/archive/full_trajectory_history.md)、[失败变体](../experiments/flamingo_map_reader/results/archive/abandoned_readout_variants.md)。归档保留当时语境，不维护当前进度。
