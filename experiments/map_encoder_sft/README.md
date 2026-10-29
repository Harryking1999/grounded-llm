# 地图编码器与 LLM 读取

固定5类版的50,000条轨迹预算，用当前采用的Qmap重新生成并认证完整轨迹、向量与排序，比较地图特征提取、cross-attention读取方式和主干训练范围。实验定义见[DESIGN](DESIGN.md)，正式参数与首批顺序见[配置](configs/five_task_50k.json)。

## 路径与实现

| 路径 | 职责 |
|---|---|
| `configs/five_task_50k.json` | 4层8组、数据与训练预算、首批GPU分配；2层只登记为备用 |
| `src/prepare.py` | 优先保留起终点分布，使用新地图生成认证路线、统一目标坐标、匹配难度与覆盖 |
| [core/reader/contextual.py](../../grounded_llm/core/reader/contextual.py) | 共同／配对特征编码；保持当前、目标与全部候选输出 |
| [core/reader/build.py](../../grounded_llm/core/reader/build.py) | 根据配置构造encoder与既有cross-attention |
| [任务适配](../../grounded_llm/tasks/state_graph_sft.py) | 当前、目标和全部候选的目标坐标转换与在线评测 |
| [原SFT入口](../map_sft_baselines/src/train.py) | 复用数据重编号、公共训练、恢复与保存 |
| [原队列](../map_sft_baselines/src/queue.py)、[bridge](../map_sft_baselines/src/bridge.py) | 两卡训练lane、028／4090共享分片评测与原早停规则 |
| `tests/test_contextual_reader.py` | 候选对应、padding、配对隔离、时间线、冻结梯度与checkpoint读取 |

所有baseline分别从同一基座开始，共用一份按新地图重建的50k数据。原5类来源为028的 `data/flan_generalization_6f2af92/categories5`；原数据与旧结果保持原条件。

源码通过已提交归档部署到028和4090的 `code/map_encoder_<commit>/`。028使用 `data/map_encoder_<commit>/` 和 `runs/map_encoder_<commit>/`；4090仅保存对应评测数据、受限推理缓存和答卷。连接与Python入口见[机器手册](../../DEVELOPMENT_MACHINES.md)。PID、实时GPU、日志与进度只在上述忽略的run及本地 `runs/remote_status/` 中维护。

## 运行入口

```bash
python -m experiments.map_encoder_sft.src.prepare --config experiments/map_encoder_sft/configs/five_task_50k.json --source ORIGINAL_FIVE_TASK_DATA --sources ORIGINAL_SOURCE_PATHS --out NEW_DATA --model-path MODEL --workers 8
python -m experiments.map_sft_baselines.src.queue train --config experiments/map_encoder_sft/configs/five_task_50k.json --data NEW_DATA --model-path MODEL --root RUN
python -m experiments.map_sft_baselines.src.queue evaluate --config CONFIG --data NEW_DATA --model-path MODEL --root RUN --q-checkpoint NEW_DATA/blocks/qmap.pt --gpus 6
```

跨机评测使用原bridge与共同分片authority，避免重复领取；实际路径映射、认证和缓存限额由忽略的runtime profile提供。两层备用不会自动加入训练队列。
