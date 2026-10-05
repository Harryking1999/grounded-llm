# 项目状态与有序 TODO

更新：2026-10-05。结果快照为北京时间 18:18；两节点队列交接后已检查。本页维护当前进度，设置见 [DESIGN](../experiments/flamingo_map_reader/DESIGN.md)，完整成绩见 [report](../experiments/flamingo_map_reader/results/report.md)。

## 当前主线

通过门控 cross-attention，让冻结 LLM 读取冻结地图并逐步行动。地址 K 与状态 V 分离；完整轨迹同时监督排序、动作、停止和总结。寻路只用一张图，积木用 1000 张训练初始棋盘；积木仅输出最近 Top-10 加 current，输入仍给全部合法候选。

两任务训练候选均覆盖全部具体动作；这不表示覆盖全部状态与目标组合。研究主张和边界见[研究简述](RESEARCH_BRIEF.md)。

## 已完成的结果

| validation | reference 动作可达率 | reference 最短路动作率 | rollout 到达率 | rollout 到达且最短 |
|---|---:|---:|---:|---:|
| 寻路 epoch 3（final） | 100.0% | 74.4% | 423/532（79.5%） | 70/532（13.2%） |
| 积木 epoch 3 | 84.9% | 77.2% | 187/1,100（17.0%） | 110/1,100（10.0%） |
| 积木 epoch 4 | 86.1% | 78.9% | 194/1,100（17.6%） | 117/1,100（10.6%） |

reference 两率只计作答前可解的决策轮；积木 epoch 4 为 6,409 轮。rollout 含初始即目标：非零步到达率，寻路为 78.2%，积木 epoch 4 为 9.4%。逐轮随机合法动作的可达基线分别为 100.0% 和 50.3%。

已有证据支持“动作选择学到了一部分，寻路同图新目标表现改善”。积木 reference 与闭环仍有明显差距；map／no_map 等匹配对照未齐，尚不能把收益归因于地图通路，也不能由少数 epoch 判断饱和。

## 训练与评测队列

- 寻路 3 个 epoch 已完成；积木按已确定的 5-epoch 预算续训，交接时约 96%，训练进程正常。
- 寻路 epoch 1–3、积木 epoch 1–4 的 validation reference／rollout 已收齐。
- 评测主节点：4 张卡运行寻路 final test reference，之后继续其余已排定测试与 no_map。
- 训练节点：1 张卡续训积木，另外 3 张卡运行寻路 final test rollout；积木真正的 final 发布后，由该节点的评测 worker 接续其 validation、test 与 no_map。
- 已修正委派队列只等 `checkpoint-245000`、接不到 `final` 的衔接问题。两队列使用互斥前缀和独立状态文件，原有在跑分片未中断；交接检查未见失败。

主队列状态为 `queue_status.json`，worker 为 `delegate_status.json`；日志在运行目录的 `logs/`。旧 `completion.json` 不是重启后整轮已完成的证明。积木 step-128 训练内诊断沿用此前停排决定，不补跑；epoch 对比使用完整整数 epoch 结果。

## 运行与源码定位

| 对象 | 标识 |
|---|---|
| 当前运行 | `long_f73b700_20261004`，两节点共享存储 |
| 完整轨迹训练协议 | `f73b700`；积木预算延长与恢复兼容修订为 `d3d9cbc`、`d9ed3c2` |
| 接管后的队列源码 | `8e62133`；不改模型、数据或训练预算 |
| 冻结积木编码器 | `tree_1000_132f5a1/best.pt`，来源说明见[积木地图](BLOCKS_QMAP_PLAN.md) |
| 证据与分母 | [long_trajectory_summary.json](../experiments/flamingo_map_reader/results/long_trajectory_summary.json) |

连接信息、实际运行路径和接管命令保存在本地忽略文件 `runs/map_reader_handoff.json`，远端启动记录为运行目录内的 `handoff_owner.json`／`handoff_worker.json`。不把端口和进程号当作实验身份。目录与汇总命令见[实验 README](../experiments/flamingo_map_reader/README.md)。

## 有序 TODO

1. 完成既有积木第 5 个 epoch 及其 validation；保留 epoch 3／4 比较，不将旧 epoch 3 的过渡 final 当作当前最终权重。
2. 收齐两任务 final test 与同权重 no_map。按同题同编号配对，以逐例计数汇总，不平均分片百分比。
3. 补全重编号一致性、真实目标配对；报告旧可达性字段的覆盖率，缺失不补零。两类积木留出组及零步组分开报告。
4. 利用已有轨迹定位积木首次走入死局或非法动作的位置，区分地图排序、接口选择和自身历史影响，再决定是否调整训练。暂不启动新实验条件。

## 历史证据入口

- 地图读取：[旧完整轨迹与小样本](../experiments/flamingo_map_reader/results/archive/full_trajectory_history.md)、[swap](../experiments/flamingo_map_reader/results/archive/swap_training.md)、[失败变体](../experiments/flamingo_map_reader/results/archive/abandoned_readout_variants.md)。
- 地图与基线：[实验目录](../experiments/README.md)、[积木地图来源](BLOCKS_QMAP_PLAN.md)。
- 旧连续 token 路线：[历史计划](archive/EXPERIMENT_TODO.md)、[路线备忘](archive/PROJECT_ROADMAP.md)。归档只作历史参考，不维护当前进度。
