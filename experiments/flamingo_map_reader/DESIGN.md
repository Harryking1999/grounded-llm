# 积木分离 K/V：重新训练

[2026-10-08 会议决议](../../docs/meetings/2026-10-08.md)及用户补充为当前依据。停止融合 FFN 运行，复用既有物理任务、标签、地图与语言基座，从头初始化原分离 K/V 接口。

## 架构与代码来源

复用 `memory.py` 的 `AddressedMapMemoryEncoder`、`fusion.py` 的门控 cross-attention 和 `train.py` 的 `build_reader`。分离接口最初由提交 `1971619` 引入，上一轮配置为 `blocks1000_long.json`；当前实现保留了该路径，未复制一套旧源码。

K 仅由角色和候选编号 embedding 拼接构成；V 为冻结地图状态向量的线性投影。没有前置融合 FFN，也没有状态残差 FFN。语言隐状态提供 Query，各层经门控将地图读取结果加入语言残差。冻结 Qwen2.5-1.5B-Instruct 和地图，只训练原有接口、角色/编号 embedding 及门控。每轮文本只能读取对应地图快照。

## 数据与训练

正式合同：[blocks_kv_restart.json](configs/blocks_kv_restart.json)。复用上一轮的已缓存数据，通过 `--prepared-manifest` 创建新清单和只读数据目录链接，不加载旧 adapter、optimizer 或 scheduler。

物理任务仍为普通成功轨迹、初始即目标、失败上下文三类。仅普通轨迹保留旧编号增强；两个终止类别不增强。清单限制失败样本仅使用第一个已缓存编号版本，既有缓存保持原样。用户已明确两个终止类别保持一致；因此不继续沿用旧运行的失败样本增强。

失败上下文仍只监督末轮无解答案，之前 assistant 回答仅作为上下文。无解条件为尚未达标且没有合法动作，不要求中途识别全局死局。输出沿用：

```text
No solution: no legal moves remain and the goal has not been reached.
<action>none</action>
<done/>
```

普通轨迹继续监督 Top-10 加 current 的地图距离排序和动作；初始即目标监督正确停止。输入包含全部合法候选，不提供距离数值或答案排序。提示、目标标签、冻结地图来源及损失沿用既有实现。

训练从零开始，四卡 DDP，每卡一条轨迹；每条轨迹按监督 token 平均，再跨卡等权平均。沿用逐层重计算和分块交叉熵以控制显存。学习率、warmup、训练预算、保存及生成预算以正式配置为准，不在说明中复制机器参数。完整断点可在同一运行目录恢复；新运行不能续接融合 FFN 权重。

## 评测与指标

正式口径：[blocks_kv_evaluation.json](configs/blocks_kv_evaluation.json)。普通任务按最短长度分层，两类棋盘来源均衡；分别运行 reference（示范历史）和 rollout（自身历史）。已达标与固定失败上下文另行评测，三类任务分别给分子和分母，不做跨任务加权或混池指标。

- 总体路径指标：普通任务 rollout 完整路径到达率；到达且正确停止另列。
- 普通任务：单步保持目标可达率、地图最优动作率、两种 NDCG@10，以及各自对应的 NDCG@1。
- 已达标任务：正确到达判断与停止率。
- 无解任务：正确无解输出率；固定失败历史和自身 rollout 耗尽分开报告。

NDCG 使用线性 gain 与标准 `1/log2(i+1)` 位置折扣。第一种 gain 为地图真实排序位置倒数 `1/r`，真实距离并列时取其所占位置 gain 的均值；第二种按用户 2026-10-09 决定改为状态内 Q-map 距离归一化 `(d_max-d)/(d_max-d_min)`，最小和最大距离取自该状态全部候选。距离全相等（含单候选）时所有 gain 为 1，这些状态另行计数。@1 为回答首位候选 gain 与真正最优 gain 的比值；首位并列取平均 gain。current 不参加候选 NDCG；无法解析计零，缺失位置计零。同状态精确随机期望与全部候选从远到近的反序作为基线；反序 @10 取最远的前 10 个。两种 NDCG 评价地图排序，不等同于真实可达性。

地图排序以全部未达标且有候选的状态为分母，含 rollout 已走死的状态；单步可达率只在动作前仍可解的状态计分，非法动作和提前停止计错。另报逐轮结果以暴露早期错误。原始评测保留旧 rank-grade 与距离倒数 NDCG 兼容字段；正式汇总使用 `ndcg_inverse_rank_at_*` 与 `ndcg_minmax_distance_at_*`，旧回答通过 CPU 重评分转换，不将旧字段改名冒充新分数，不修改历史报告的定义。

## 运行与归档

统一训练入口 `retrain_blocks`，传入新配置、旧 prepared manifest、基座路径和独立输出目录，并使用 `--training-only`。另一节点使用 `blocks_checkpoint_queue --terminal-tasks --cache-map-kv` 评测所有三类任务。汇总时向 `summarize_blocks_results` 传入本轮评测配置。

运行路径、节点、PID、日志与启动检查保存到 Git 忽略的 `runs/`。当前结果见 [report](results/report.md)。

融合 FFN 运行已停止。其 [设计](DESIGN_ffn_failure.md)、[0.5–5 epoch 报告](results/report_ffn_failure.md)与[紧凑结果](results/blocks_ffn_failure_summary.json)保留；原始回答、断点、日志和数据留在原运行目录，避免破坏现有符号链接。停机时约 5.36 epoch，正式对照截至已全量评齐的 epoch 5。更早分离 K/V 的两任务证据见 [上一轮报告](results/report_long_trajectory.md)。这些运行数据、样本权重及预算不同，不构成纯架构因果对照。
