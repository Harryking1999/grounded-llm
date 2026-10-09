# Cross-attention 地图读取

冻结语言模型和地图，训练读取接口，让 LLM 输出地图排序、动作与终止判断。当前积木按 2026-10-08 会议重新训练：恢复分离 K/V，取消 FFN，仅普通任务保留编号增强，两个终止类别不增强。

- [当前设计](DESIGN.md)、[训练合同](configs/blocks_kv_restart.json)、[评测合同](configs/blocks_kv_evaluation.json)、[当前结果](results/report.md)。
- 已完成本轮 10 epoch，继续训练至 20 epoch 的[预算](configs/blocks_kv_twenty_epoch_extension.json)与[两节点安排](configs/blocks_kv_two_node_continuation.json)单独保存，原训练与数据合同保留。
- [项目状态与 TODO](../../docs/PROJECT_STATUS_AND_TODO.md)为唯一当前状态页。
- 已停止融合 FFN 的[设计](DESIGN_ffn_failure.md)、[epoch 5 报告](results/report_ffn_failure.md)及[结果摘要](results/blocks_ffn_failure_summary.json)。
- 原分离 K/V 两任务[设计](DESIGN_long_trajectory.md)与[报告](results/report_long_trajectory.md)。

## 代码与启动

原代码仍在当前权威接口：`memory.py::AddressedMapMemoryEncoder` 提供角色/编号 K 与状态 V；`fusion.py` 提供门控读取；`train.py::build_reader` 按配置选择架构。没有另建旧源码副本。数据、提示与评分复用 `trajectory_dataset.py`、`blocks_prompt.py`、`trajectory_metrics.py`。

```bash
python -m experiments.flamingo_map_reader.src.retrain_blocks \
  --config experiments/flamingo_map_reader/configs/blocks_kv_restart.json \
  --prepared-manifest "$PREPARED_MANIFEST" --source-manifest "$SOURCE_MANIFEST" \
  --model-path "$MODEL_PATH" --out "$RUN_DIR" --training-only
```

新目录从零初始化接口，复用缓存而不加载旧权重。仅限制失败样本清单的编号版本数量，原缓存不改动。训练节点使用配置指定的四卡，另一节点两卡评测：

```bash
python -m experiments.flamingo_map_reader.src.blocks_checkpoint_queue \
  --run "$RUN_DIR" --model-path "$MODEL_PATH" --gpus 0,1 \
  --terminal-tasks --cache-map-kv
```

每半 epoch 评测普通 reference/rollout、初始目标及失败上下文。原始回答和分片摘要保存在 `evaluation_half_epoch/step-N/`；未完成分片先归档再重做，完整分片复用。运行中不得同时启动重叠分片。

完成原预算后的续训使用 `train --resume --resume-topology --epoch-extension`，继续原 optimizer、scheduler 与样本进度，新增权重写入 `training_extension_20epoch/models/`。评测队列通过 `--continuation-models` 同时扫描新增权重并等待新 final；原数据 manifest 与原 `training/models/final` 保留。机器、命令参数和运行目录见两节点合同与开发机手册。

```bash
python -m experiments.flamingo_map_reader.src.summarize_blocks_results \
  --run "$RUN_DIR" --config experiments/flamingo_map_reader/configs/blocks_kv_evaluation.json \
  --out experiments/flamingo_map_reader/results/blocks_results_summary.json \
  --ranking-cache "$RUN_DIR/diagnostics/ranking_cache.json"
```

普通任务报告完整到达、单步可达、地图最优及位置倒数／状态内距离归一化两种 gain 的 NDCG；距离归一化已替代距离倒数，随机与反序基线使用同一组状态。两个终止任务分别报告。分片原始汇总仍保留旧兼容字段，CPU 重评分后正式报告使用本轮合同，不跨任务混算。

纯地图同题对照使用 `evaluate_blocks_map_baselines` 和[合同](configs/blocks_map_baseline_comparison.json)，直接复用当前 manifest 的 510 道 test 任务、规则、动作预算和随机种子；CPU 重跑原图并逐题核对缓存后，与新图直接贪心比较。现用图 345/510，新图 397/510，逐题回退与分组结果保存在[结果摘要](results/blocks_results_summary.json)的 `map_greedy_comparison`；此次地图对照没有更换 LLM 的地图输入或训练数据。

模型、地图、数据、日志和运行状态留在 Git 外；以代码提交、配置与运行目录识别产物。停止的融合 FFN 目录保留原位并标记 archived，避免破坏缓存链接。历史诊断及旧入口说明见归档设计和报告。
