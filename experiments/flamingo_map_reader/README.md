# Cross-attention 地图读取

冻结 Qwen2.5-1.5B-Instruct 和地图，只训练读取接口，让 LLM 在完整轨迹中输出远近排序、动作、停止与总结。寻路沿用角色／编号作 K、状态内容作 V 的方案；积木本次通过共享 FFN 融合地图、角色及编号，同时供 K/V 使用，并加入失败上下文。积木答案限最近 Top-10 加 current，输入保留全部候选。

## 阅读顺序

1. [DESIGN.md](DESIGN.md)：数据、输入文本、监督、训练和评测口径。
2. [上一轮 DESIGN](DESIGN_long_trajectory.md) 与[上一轮 report](results/report_long_trajectory.md)：原两任务的协议、成绩和辅助诊断，均在原目录保留。
3. [项目状态与 TODO](../../docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页。

当前正式配置为 [path_single_long.json](configs/path_single_long.json) 和 [blocks_ffn_failure_batch4.json](configs/blocks_ffn_failure_batch4.json)；全局 batch 64 对照保留在 [blocks_ffn_failure.json](configs/blocks_ffn_failure.json)，上一轮积木配置为 [blocks1000_long.json](configs/blocks1000_long.json)。[上一轮结果摘要](results/long_trajectory_summary.json)保存其分片来源、计数和数据统计；模型、原始回答及运行日志留在 Git 外。

## 积木会议决策重训

新一轮积木合同见 [blocks_ffn_failure_batch4.json](configs/blocks_ffn_failure_batch4.json)，入口为
[retrain_blocks.py](src/retrain_blocks.py)。冻结原语言模型和地图，从头初始化读取接口；
用户最终选择融合版：128 维地图状态与 32 维角色、32 维编号 embedding 拼接，
经单个共享 FFN（192→1024→GELU→256）输出融合特征，再供各读取层独立的 K/V 投影使用。
K 和 V 均携带地图与角色信息；FFN 在所有槽位和读取层间共享。语言 Query 仍来自当前层文本隐状态。
这是本项目的地图特征提取器，并非对 Flamingo 原论文全部结构的复现。

2026-10-07 用户停止全局 batch 64 的运行，要求仅改为每卡 batch 1、四卡全局 batch 4，
从相同随机种子重新训练。复用同一份准备数据、编号和标签，架构及其余训练设置不变；
每 0.1 epoch 保存完整断点，另在评测节点按 0.5 epoch 间隔评测。不同 batch 使用独立输出目录，
不直接续接另一 batch 的 optimizer 或数据位置。此对照同时改变更新频率与 batch 内 token 平均产生的样本权重，
不能仅凭它把效果变化归因于更新次数。

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
正式启动按合同验证所选 batch，当前配置只允许每卡 1、全局 4，
随后训练并自动评测 final checkpoint。旧配置和旧结果保留用于对照，不混入这次成绩。
最长样本的旧实现连全局 batch 8 都会显存不足，因此训练启用逐层激活重计算，
并仅对有监督 token 分块计算交叉熵；每卡先按本卡监督 token 求平均，再平均四卡梯度。
同一个 batch 上的普通计算与优化计算已比较 loss 和全部可训练梯度；这不意味着不同 batch 的样本权重相同。

从仓库根目录启动完整流程（路径参数由调用者指定，节点信息不写入源码）：

```bash
python -m experiments.flamingo_map_reader.src.retrain_blocks \
  --config experiments/flamingo_map_reader/configs/blocks_ffn_failure_batch4.json \
  --prepared-manifest "$PREPARED_MANIFEST" \
  --source-manifest "$SOURCE_MANIFEST" --model-path "$MODEL_PATH" --out "$RUN_DIR"
```

本轮从头对照须加 `--prepared-manifest "$PREPARED_MANIFEST"`，复用已停止大 batch 运行的轨迹和编号，
并为 `RUN_DIR` 选择新目录。该选项只复用数据，不加载旧权重。保存间隔由合同的 `checkpoint.every_epoch_fraction` 控制：
当前每 1,375 steps（0.1 epoch）保存，10 epochs 共 137,500 steps。
中断后，以同一配置、模型路径和输出目录调用上述命令并加 `--resume`；
入口复用现有队列的断点选择及评测分片重试，底层仍调用 `train.py --resume CHECKPOINT`。
只选择已发布且包含 adapter、optimizer、scheduler、trainer state 和全部进程 RNG 状态的断点；
初始权重与未写完的目录不作为续训断点，无完整断点时明确报错。
Trainer 恢复已完成步数、epoch 与数据顺序，继续原有总预算，不额外增加训练轮数。
必须沿用同一运行的 batch 和设备拓扑；未保存的更新从最近完整断点重算。不同 batch 的运行不直接互相续训。

训练结束后复用 `trajectory_eval.py` 完成主测试 reference、主测试 rollout 和失败上下文 reference，
由四个 GPU 分片执行，再用同一 `aggregate` 汇总；完成的评测分片在恢复时直接复用。
也可独立调用任意已保存权重进行评测：

```bash
python -m experiments.flamingo_map_reader.src.trajectory_eval \
  --manifest "$RUN_DIR/data/manifest.json" --model-path "$MODEL_PATH" \
  --adapter-checkpoint "$CHECKPOINT_DIR/adapter.pt" \
  --split test --mode reference --variants 1 --out "$EVAL_DIR"
```

`--mode rollout` 用于自主执行，`--split test_no_solution --mode reference` 用于失败上下文诊断。
分片参数 `--start`、`--stop` 与生成预算均沿用共享评测器及合同，原始回答和摘要留在运行目录。

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

当前积木独立运行在根目录的 `data/` 保存清单与轨迹、`training/` 保存合同与权重，
`evaluation/<split>_<mode>_<start>_<stop>/` 保存 final 分片，`evaluation_half_epoch/step-N/test_<mode>/<start>_<stop>/` 保存中间评测分片。
上一轮两任务队列则使用 `<task>/data/`、`<task>/training/` 和 `evaluation/<task>/<checkpoint>/<split>_<mode>_<map|no_map>/`。
以提交、合同和运行路径识别实验，不以分支名判断设置。

## 两节点协作与汇总

**当前积木：**训练节点运行 `retrain_blocks`；另一节点另行启动半 epoch 评测队列，读取同一共享目录中
已发布的 checkpoint，两卡 rollout、两卡 reference，优先最新节点后补旧节点。该队列不由训练入口自动拉起；
运行源码快照中的 `blocks_half_epoch_eval_queue.py` 和 `evaluation_half_epoch/launch.json` 保留调用，
`contract.json` 记录分片合同，`status.json` 记录排队状态。首次评测在 0.5 epoch（step 6,875）发布后启动。
中间队列只评主测试，独立无解诊断由 final 流程执行。重启跳过完整分片，保留未完成输出后整片重做，
汇总排除 `.interrupted.*` 目录；同一 final 权重若被两条流程评测，也不能算作独立重复。详见 [DESIGN 第 7 节](DESIGN.md#7-评测设置与指标口径)。

**上一轮两任务队列：**以下 `--delegate-prefix`、`--evaluation-only` 参数属于 `trajectory_queue`，不适用于当前 `retrain_blocks` 入口。

两节点共享运行目录。主队列用 `--delegate-prefix` 排除交给另一节点的任务；另一节点运行同一模块的 `--evaluation-only --include-prefix ... --status-file ...`，使用独立状态文件。前缀必须互斥，不能将同一分片交给两个进程。

评测 worker 只读取已发布 checkpoint，等待真正的 `final` 后完成最终评测。重启时接管仍在运行的子进程，跳过已有分片摘要；主队列和 worker 的完成状态分别记录。节点地址和进程号只记在忽略的运行记录中。

从仓库根目录重建上一轮两任务证据：

```bash
python -m experiments.flamingo_map_reader.src.collect_evidence \
  --run-root RUN --update experiments/flamingo_map_reader/results/long_trajectory_summary.json --table
```

只汇总已完成分片，同一共享路径计一次；未完成条件不能当作全量成绩。回填工具只在确有旧字段缺失时使用，不重跑模型生成。

历史入口：[完整轨迹旧实验](results/archive/full_trajectory_history.md)、[swap](results/archive/swap_training.md)、[失败变体](results/archive/abandoned_readout_variants.md)。旧配置与代码保留复现，不作为当前运行入口。
