# 地图读取接口：寻路与积木

两项任务共用 Qwen2.5-1.5B-Instruct 的基座型号、地图记忆编码与门控 cross-attention 的结构和代码、整题 SFT 和逐轮文字协议。**寻路与积木分别新建并训练 P、cross-attention 等接口参数，保存到不同 checkpoint；不混合训练，也不共享训练后的接口权重。**地图与环境规则各自提供。设计与解释见 [DESIGN](DESIGN.md)，进度见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

正式合同：[寻路](configs/pilot_path256.json)、[积木](configs/pilot_blocks.json)。`src/text.py`、`src/sft.py` 统一英文开头、排序和终止回答；`src/train.py` 根据合同加载图 Q/V 或共享棋盘 Q。训练使用普通 Trainer batch、assistant-only CE，保持每轮 token 到当轮地图的方案 A 绑定。两个任务的训练超参数相同；实际 batch 由各自长题的显存测量决定。

## 数据与运行入口

以下路径由部署环境提供；所有 `runs/` 产物位于 Git 外。生成正式数据前，需让命令与已提交合同一致。

```bash
python -m experiments.flamingo_map_reader.src.prepare \
  --config experiments/flamingo_map_reader/configs/pilot_path256.json \
  --source-root GRAPH_SOURCE_ROOT \
  --output runs/flamingo_map_reader/data/path_manifest.json

python -m experiments.flamingo_map_reader.src.blocks_data \
  --config experiments/flamingo_map_reader/configs/pilot_blocks.json \
  --official-data OFFICIAL_H5 --q-checkpoint Q_BEST_PT \
  --q-training-data Q_TRAIN_DATA_NPZ \
  --output runs/flamingo_map_reader/data/blocks_manifest.json

python -m experiments.flamingo_map_reader.src.train \
  --config experiments/flamingo_map_reader/configs/pilot_path256.json \
  --manifest runs/flamingo_map_reader/data/path_manifest.json \
  --source-root GRAPH_SOURCE_ROOT --batch-size MEASURED_BATCH \
  --out runs/flamingo_map_reader/path_train

python -m experiments.flamingo_map_reader.src.train \
  --config experiments/flamingo_map_reader/configs/pilot_blocks.json \
  --manifest runs/flamingo_map_reader/data/blocks_manifest.json \
  --q-checkpoint Q_BEST_PT --batch-size MEASURED_BATCH \
  --out runs/flamingo_map_reader/blocks_train
```

续训在同一训练命令上添加 `--resume .../models/checkpoint-N`。图闭环评测入口为 `src.evaluate_graph`，积木为 `src.evaluate_blocks`；两者都读取对应 manifest、adapter checkpoint 与原始地图。结果保存逐题实际动作和模型回答及紧凑 `results/summary.json`。图评测使用保留的原五图题集，积木按三组未见状态统计。

寻路另有独立的首步地图关系评测，使用[读出门槛](configs/path_readout_gate.json)。它在 64 条验证题和五图 120 条保留题的固定首状态生成一次回答，从冻结 Q/V 重新计算 `current` 与每个候选的远近，报告排序格式、逐对关系、完整排序、最近候选，以及动作是否遵循排序。验证题沿用训练示范的首轮文本和候选编号；保留题沿用闭环评测的首轮文本和编号。两个集合分别满足门槛，才称当前权重通过首步距离关系读出；这仍不代表整题规划成功。保留题闭环轨迹也可由 `src.summarize_graph_eval` 重放并按首轮／后续轮统计同样指标，用来定位后续状态的错误。

```bash
python -m experiments.flamingo_map_reader.src.evaluate_graph_readout \
  --config experiments/flamingo_map_reader/configs/pilot_path256.json \
  --gate-config experiments/flamingo_map_reader/configs/path_readout_gate.json \
  --manifest runs/flamingo_map_reader/data/path_manifest.json \
  --source-root GRAPH_SOURCE_ROOT --adapter-checkpoint PATH_ADAPTER_PT \
  --out runs/flamingo_map_reader/path_readout
```

读出分数是对模型文字的检验；如果分数高，还需交换候选 Q 向量而保持文字不变，检查选择是否随地图变化，才能支持“使用了地图”这一因果解释。

最小正确性检查：

```bash
python -m unittest discover -s experiments/flamingo_map_reader/tests -p 'test_*.py'
```
