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

完成原预算后的续训使用 `train --resume --epoch-extension`，继续原 optimizer、scheduler 与样本进度，新增权重写入 `training_extension_20epoch/models/`；仅在明确的四卡每卡一条迁移到两卡每卡两条时使用 `--resume-topology`。当前按确认的原四卡每卡一条续训。评测队列通过 `--continuation-models` 同时扫描新增权重并等待新 final；原数据 manifest 与原 `training/models/final` 保留。机器、命令参数和运行目录见平台两节点合同与开发机手册。

单卡辅助队列串行处理 reference 和 rollout；与平台主队列使用同一分片合同并持有互斥的余数。跨机器复制的权重通过 `--source-path-map` 显式映射 manifest、基座与地图位置，配置和任务记录仍严格匹配；逐题输出回传到平台后由主队列统一汇总。

续训和协同评测的日常启动使用 `experiment_runner`，直接读取同一份安排合同，不再复制每轮启动脚本。在执行节点准备 Git 外的 runtime profile，仅包含 `workspace_root`、`code_root`、`python`、`model_path`；训练增加 `q_checkpoint`，迁移后的评测增加 `source_path_map`，可选 `log`。路径必须是该节点上的绝对路径，profile 保存在 `runs/<run>/transfer/` 或 `tmp/`，不保存密码。

```bash
python -m experiments.flamingo_map_reader.src.experiment_runner plan \
  --contract "$CONTRACT" --profile "$RUNTIME_PROFILE" --role training
# 同一入口：start 后台启动，status 查看状态；role 也可取 evaluation / auxiliary_evaluation。
```

`plan` 不启动任务；`start` 在 Linux 节点后台调用现有 `train` / `blocks_checkpoint_queue`，记录退出状态，检测现有同目录进程并拒绝重复启动。训练 batch 与 GPU 数必须满足合同；改变拓扑仍需独立的正式迁移合同。已运行的旧入口任务可以用 `status` 读取原状态，无需重启。

不共享目录时，传输工具 `artifact_bridge` 独立运行于产物所在的源节点。其 runtime profile 包含 `source_workspace`、`target_workspace`、`jump` / `target`（各有 `host`、`port`、`user`）、`stage_dir`、`known_hosts`、`status` 和可选 `interval_seconds`。正式分片数量和归属仍读取同一个安排合同。基座、地图、数据缓存和显式路径映射须先准备好；工具不安装环境、不启动或重启 GPU 队列。

```bash
python -m experiments.flamingo_map_reader.src.artifact_bridge \
  --contract "$CONTRACT" --profile "$TRANSFER_PROFILE" --detach
# --once 只同步一轮；默认每 600 秒同步，完成后退出，失败写状态并退出。
```

密码通过交互提示读入，仅留在内存；后台子进程通过 stdin 接收，不放在命令行、配置或日志里。SSH 主机密钥必须预先核对。新半 epoch adapter 经跳板中转，最后发布可评测标记；同步主节点完成摘要以支持远端队列结束判断，返回完整逐题输出后才发布摘要。结果汇总仍复用下述现有入口。两种工具的日志、状态、锁、中转包和机器 profile 均留在 Git 外。

```bash
python -m experiments.flamingo_map_reader.src.summarize_blocks_results \
  --run "$RUN_DIR" --config experiments/flamingo_map_reader/configs/blocks_kv_evaluation.json \
  --out experiments/flamingo_map_reader/results/blocks_results_summary.json \
  --ranking-cache "$RUN_DIR/diagnostics/ranking_cache.json"
```

普通任务报告完整到达、单步可达、地图最优及位置倒数／状态内距离归一化两种 gain 的 NDCG；距离归一化已替代距离倒数，随机与反序基线使用同一组状态。两个终止任务分别报告。分片原始汇总仍保留旧兼容字段，CPU 重评分后正式报告使用本轮合同，不跨任务混算。

纯地图同题对照使用 `evaluate_blocks_map_baselines` 和[合同](configs/blocks_map_baseline_comparison.json)，直接复用当前 manifest 的 510 道 test 任务、规则、动作预算和随机种子；CPU 重跑原图并逐题核对缓存后，与新图直接贪心比较。现用图 345/510，新图 397/510，逐题回退与分组结果保存在[结果摘要](results/blocks_results_summary.json)的 `map_greedy_comparison`；此次地图对照没有更换 LLM 的地图输入或训练数据。

模型、地图、数据、日志和运行状态留在 Git 外；以代码提交、配置与运行目录识别产物。停止的融合 FFN 目录保留原位并标记 archived，避免破坏缓存链接。历史诊断及旧入口说明见归档设计和报告。
