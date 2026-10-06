# Cross-attention 地图读取

冻结 Qwen2.5-1.5B-Instruct 和地图，只训练读取接口，让 LLM 在完整轨迹中输出远近排序、动作、停止与总结。当前条件是单图寻路和 1000 初始棋盘积木；角色／编号作 K，状态内容作 V。积木答案限最近 Top-10 加 current，输入保留全部候选。

## 阅读顺序

1. [DESIGN.md](DESIGN.md)：数据、输入文本、监督、训练和评测口径。
2. [results/report.md](results/report.md)：reference 与 rollout 主结果、地图对照和辅助诊断。
3. [项目状态与 TODO](../../docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页。

正式配置为 [path_single_long.json](configs/path_single_long.json) 和 [blocks1000_long.json](configs/blocks1000_long.json)。[结果摘要](results/long_trajectory_summary.json)保存分片来源、计数和数据统计；模型、原始回答及运行日志留在 Git 外。

## 积木会议决策重训

新一轮积木合同见 [blocks_ffn_failure.json](configs/blocks_ffn_failure.json)，入口为
[retrain_blocks.py](src/retrain_blocks.py)。冻结原语言模型和地图，从头初始化读取接口；
用户最终选择融合版：128 维地图状态与 32 维角色、32 维编号 embedding 拼接，
经单个共享 FFN（192→1024→GELU→256）输出融合特征，再供各读取层独立的 K/V 投影使用。
K 和 V 均携带地图与角色信息；FFN 在所有槽位和读取层间共享。语言 Query 仍来自当前层文本隐状态。
这是本项目的地图特征提取器，并非对 Flamingo 原论文全部结构的复现。

保留原 9000 条成功训练轨迹，另加 1000 条地图贪心失败轨迹。旧数据清单没有失败训练样本，
原采样器记录剔除了 3282 条贪心失败样本但未保存具体轨迹，因此在训练棋盘上重新生成。
新增样本照常执行地图贪心动作，直到没有合法动作且仍未到目标；不要求中途判断不可达。
之前所有 assistant 回答均为上下文、loss 为零，仅最后无解回答参与监督。
无解终止仅依赖当前棋盘与合法候选列表；oracle 仍用于任务最短长度和原有评测指标。
用户确认使用 action none 和 done；按后续修订的终止条件，固定答案为：

```text
No solution: no legal moves remain and the goal has not been reached.
<action>none</action>
<done/>
```

主测试集按最短解长度分层，两个分组各占一半；另留独立失败上下文测试，报告无解召回率。
主测试的 reference 和 rollout 同时报告误判无解率；走错后正确识别无解仍不算完成原任务。
可先用 [find_training_batch.py](src/find_training_batch.py) 扫描训练集全部编号变体，
将序列长度、地图快照数、候选槽位数和监督长度的极值样本组合，
以真实四卡反向传播和优化器更新测量 batch 显存边界及保留余量的档位。
正式启动再按合同从大到小验证所选全局 batch，
随后训练并自动评测 final checkpoint。旧配置和旧结果保留用于对照，不混入这次成绩。
最长样本的旧实现连全局 batch 8 都会显存不足，因此训练启用逐层激活重计算，
并仅对有监督 token 分块计算交叉熵；目标仍是同一 assistant-token 平均交叉熵。
单元测试比较普通计算与优化计算的 loss 和全部可训练梯度，避免因节省显存改变监督。

## 代码与目录

| 功能 | 入口 |
|---|---|
| 物理任务、划分与示范 | [prepare_trajectories.py](src/prepare_trajectories.py)、[sft.py](src/sft.py)、[blocks_sft.py](src/blocks_sft.py) |
| 文本与逐轮地图绑定 | [prompt.py](src/prompt.py)、[blocks_prompt.py](src/blocks_prompt.py)、[transcript.py](src/transcript.py) |
| K/V 分离与门控读取 | [memory.py](src/memory.py)、[fusion.py](src/fusion.py) |
| 编号增强、缓存与训练 | [trajectory_protocol.py](src/trajectory_protocol.py)、[trajectory_dataset.py](src/trajectory_dataset.py)、[train.py](src/train.py) |
| 自由生成与评分 | [trajectory_eval.py](src/trajectory_eval.py)、[trajectory_metrics.py](src/trajectory_metrics.py) |
| 训练／评测调度 | [trajectory_queue.py](src/trajectory_queue.py) |
| 结果汇总与旧字段回填 | [collect_evidence.py](src/collect_evidence.py)、[backfill_reachability.py](src/backfill_reachability.py) |

积木地图来自 `tree_1000_132f5a1/best.pt`，源码来源为 `132f5a1`，依赖 [blocks_distance_map](../blocks_distance_map/README.md) 的编码器、棋盘读取与裁判。寻路复用 [external_map_interface](../external_map_interface/README.md) 的图环境和 Q/V。

运行根目录内，`<task>/data/` 保存清单与轨迹，`<task>/training/` 保存训练合同和权重，`evaluation/<task>/<checkpoint>/<split>_<mode>_<map|no_map>/` 保存评测分片。以提交、合同和运行路径识别实验，不以分支名判断设置。

## 两节点协作与汇总

两节点共享运行目录。主队列用 `--delegate-prefix` 排除交给另一节点的任务；另一节点运行同一模块的 `--evaluation-only --include-prefix ... --status-file ...`，使用独立状态文件。前缀必须互斥，不能将同一分片交给两个进程。

评测 worker 只读取已发布 checkpoint，等待真正的 `final` 后完成最终评测。重启时接管仍在运行的子进程，跳过已有分片摘要；主队列和 worker 的完成状态分别记录。节点地址和进程号只记在忽略的运行记录中。

从仓库根目录重建证据：

```bash
python -m experiments.flamingo_map_reader.src.collect_evidence \
  --run-root RUN --update experiments/flamingo_map_reader/results/long_trajectory_summary.json --table
```

只汇总已完成分片，同一共享路径计一次；未完成条件不能当作全量成绩。回填工具只在确有旧字段缺失时使用，不重跑模型生成。

历史入口：[完整轨迹旧实验](results/archive/full_trajectory_history.md)、[swap](results/archive/swap_training.md)、[失败变体](results/archive/abandoned_readout_variants.md)。旧配置与代码保留复现，不作为当前运行入口。
