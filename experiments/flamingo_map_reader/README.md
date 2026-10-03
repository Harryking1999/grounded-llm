# 地图读取：长轨迹 SFT

按原飞书 4.2，寻路固定一张图及其 Q，积木使用 1000 张初始棋盘与共享冻结 Q。两任务分别从头训练读取接口，冻结 Qwen2.5-1.5B-Instruct 与地图。训练完整长轨迹的每次判断、距离、排序、动作及最终停止和总结。

- [设计与评价口径](DESIGN.md)
- [寻路正式配置](configs/path_single_long.json)、[积木正式配置](configs/blocks1000_long.json)
- [唯一状态页](../../docs/PROJECT_STATUS_AND_TODO.md)
- [结果报告](results/readout_failure_analysis.md)：新成绩与归档小样本分开。

## 模块职责

| 模块 | 职责 |
|---|---|
| `prepare_trajectories.py` | 长任务选择、物理任务划分、训练后缀排除、覆盖与贪心基线 |
| `trajectory_protocol.py` | 数值回答、自洽重编号；固定实际动作路径 |
| `trajectory_dataset.py` | 地图保存一次，固定编号版本的 token 缓存和重复读取 |
| `train.py` | 共用 HF Trainer，完整轨迹普通 SFT、冻结基座、checkpoint 发布 |
| `transcript.py` / `memory.py` / `fusion.py` | 全 token 的逐轮地图绑定与读取接口 |
| `trajectory_eval.py` | 两任务共用自由生成会话；参考历史下全部轮次、实际闭环、无地图 |
| `trajectory_metrics.py` | 数值、排序、动作、停止、真实历史总结评分 |
| `trajectory_smoke.py` | 最长真实样本的一次前向／反向；不保留更新后的权重 |
| `trajectory_queue.py` | 四卡资源分配、成功后推进、失败记录、checkpoint 评测和汇总 |

编号版本同步改变真实动作、后继向量、文字编号和标签，每个固定版本重复完整训练遍数。没有 Q-only swap，没有二选一预训练，也没有独立距离读出前置实验。历史入口保留复现，不从新队列调用。

## 运行

在仓库根目录，用可运行 torch / transformers 的 Python：

```bash
python -m experiments.flamingo_map_reader.src.trajectory_queue \
  --run-root /absolute/new/run \
  --model-path /absolute/Qwen2.5-1.5B-Instruct \
  --graph-source /absolute/path256_five_graphs_4ac7239 \
  --blocks-source-manifest /absolute/blocks_manifest.json \
  --blocks-official /absolute/tiling_order_10x10_8obj.h5 \
  --blocks-q /absolute/tree_1000_132f5a1/best.pt \
  --blocks-q-data /absolute/tree_1000_132f5a1/data/data.npz \
  --gpus 0 1 2 3
```

`run-root` 必须不存在。队列进度为 `queue_status.json`，日志在 `logs/`，每任务的数据和训练独立保存。准备失败、显存不足或训练失败均明确记为失败，不自动缩短任务、改变配置或覆盖运行。

每个评测分片保存逐题完整回答及分母；最后跨分片合并编号与真实目标配对。阶段诊断不替代最终全量测试，终止样本单独计数。训练集选固定物理轨迹诊断，测试集在最终 checkpoint 才解封。

针对性检查：

```bash
PYTHONPATH=.:experiments/flamingo_map_reader/tests python -m unittest \
  test_trajectory_protocol test_timeline test_trainer test_hf_integration
```

旧实验及其数值见 [完整轨迹／小样本归档](results/archive/full_trajectory_history.md) 与 [swap 归档](results/archive/swap_training.md)。
