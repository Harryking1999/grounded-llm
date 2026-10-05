# Cross-attention 地图读取

冻结 Qwen2.5-1.5B-Instruct 和地图，只训练读取接口，让 LLM 在完整轨迹中输出远近排序、动作、停止与总结。当前条件是单图寻路和 1000 初始棋盘积木；角色／编号作 K，状态内容作 V。积木答案限最近 Top-10 加 current，输入保留全部候选。

## 阅读顺序

1. [DESIGN.md](DESIGN.md)：数据、输入文本、监督、训练和评测口径。
2. [results/report.md](results/report.md)：reference 与 rollout 主结果、地图对照和辅助诊断。
3. [项目状态与 TODO](../../docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页。

正式配置为 [path_single_long.json](configs/path_single_long.json) 和 [blocks1000_long.json](configs/blocks1000_long.json)。[结果摘要](results/long_trajectory_summary.json)保存分片来源、计数和数据统计；模型、原始回答及运行日志留在 Git 外。

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
