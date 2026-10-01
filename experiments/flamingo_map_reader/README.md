# 地图读取接口：寻路与积木

两项任务共用冻结 Qwen2.5-1.5B-Instruct、地图记忆编码、门控 cross-attention、整题 SFT 和逐轮文字协议。地图与环境规则各自提供；两项分别训练和评测，不把跨任务权重共享当成已验证事实。设计与解释见 [DESIGN](DESIGN.md)，进度见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

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

最小正确性检查：

```bash
python -m unittest discover -s experiments/flamingo_map_reader/tests -p 'test_*.py'
```
