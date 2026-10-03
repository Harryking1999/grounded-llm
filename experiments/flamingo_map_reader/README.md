# 地图读取接口：寻路与积木

本实验检验冻结 Q-map 的当前、目标与合法后继向量，能否帮助冻结 Qwen2.5-1.5B-Instruct 在逐轮环境反馈中正确排序、选动作和判断终止。寻路与积木共用实现，分别训练接口、分别保存权重。

[分析报告](results/readout_failure_analysis.md)按“现象 → 训练设置 → 结果 → 解释范围”整理有效证据；[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)维护当前进度与有序 TODO；[DESIGN](DESIGN.md)定义完整任务。

## 当前保留的实验

- **原始完整轨迹 SFT：**训练投影、角色／编号和门控 cross-attention，冻结地图与 LLM；普通 assistant-token CE，保留每轮文字与对应地图。寻路训练 256 条完整轨迹，同图验证 64 条；积木按自己的完整轨迹合同生成。
- **A／B／C 小样本：**每任务 16 条完整训练轨迹、4 条留出轨迹。A 为原混合接口，B 将角色／编号只放入 K、投影状态只放入 V，C 再校准 value 尺度。均无 swap 训练；已有结果只覆盖固定首轮，不能称整条轨迹已拟合。
- **原始轨迹的两项消融：**关系 token 加权与固定 gate 均已结束，保留设置和负结果，不继续扩大。
- **评测重点：**示范轨迹各轮的独立判断与模型实际闭环各轮的判断分别看；首轮只是其中一层。完整逐轮对照尚未补齐。

已有配置是各实验的记录，不代表当前默认发起训练。参数以配置为准：

| 实验 | 配置 |
|---|---|
| 原混合接口完整轨迹 | [寻路](configs/pilot_path256.json)、[积木](configs/pilot_blocks.json) |
| K/V 解耦完整轨迹 | [寻路](configs/pilot_path256_addressed_kv.json)、[积木](configs/pilot_blocks_addressed_kv.json) |
| 小样本抽取与续训 | [抽取](configs/kv_pilot.json)、[续训](configs/kv_convergence.json)、[追加预算](configs/kv_convergence_extension.json) |
| value 尺度选择 | [校准规则](configs/kv_scale_followup.json) |
| 原始寻路的独立消融 | [关系 token ×16](configs/pilot_path256_relation_weighted.json)、[固定 gate](configs/pilot_path256_fixed_gate.json) |

历史续训与尺度合同中还列有已归档的 swap 条件；保留合同用于追溯，不能把整份历史队列作为当前待运行任务。

## 数据与运行入口

`src/text.py`、`src/sft.py` 统一逐轮排序、动作和终止回答；`src/train.py` 根据合同加载图 Q/V 或共享棋盘 Q。方案 A 让旧 token 读旧轮地图、新 token 读本轮地图，完整题作为一个训练样本。参数更新限于该任务自己的接口。

以下是完整轨迹条件的入口示例。实际模型、数据和输出路径由运行环境提供；正式运行需明确使用哪份合同。所有 `runs/` 产物位于 Git 外。

```bash
python -m experiments.flamingo_map_reader.src.prepare \
  --config experiments/flamingo_map_reader/configs/pilot_path256_addressed_kv.json \
  --source-root GRAPH_SOURCE_ROOT \
  --output runs/flamingo_map_reader/data/path_manifest.json

python -m experiments.flamingo_map_reader.src.blocks_data \
  --config experiments/flamingo_map_reader/configs/pilot_blocks_addressed_kv.json \
  --official-data OFFICIAL_H5 --q-checkpoint Q_BEST_PT \
  --q-training-data Q_TRAIN_DATA_NPZ \
  --output runs/flamingo_map_reader/data/blocks_manifest.json

python -m experiments.flamingo_map_reader.src.train \
  --config experiments/flamingo_map_reader/configs/pilot_path256_addressed_kv.json \
  --manifest runs/flamingo_map_reader/data/path_manifest.json \
  --source-root GRAPH_SOURCE_ROOT --batch-size MEASURED_BATCH \
  --out runs/flamingo_map_reader/path_train

python -m experiments.flamingo_map_reader.src.train \
  --config experiments/flamingo_map_reader/configs/pilot_blocks_addressed_kv.json \
  --manifest runs/flamingo_map_reader/data/blocks_manifest.json \
  --q-checkpoint Q_BEST_PT --batch-size MEASURED_BATCH \
  --out runs/flamingo_map_reader/blocks_train
```

`src.prepare_kv_pilot` 从冻结 manifest 生成同题小样本。A／B／C 的早期训练均为 256 步，后续按固定训练集的整体 CE 与关系 CE 判断平台、按整体 CE 选权重；各条件停止与选用步数见分析报告。排序 CE 不含末尾复制动作，仍是给定正确前文的损失，不能代替自由生成。

`88c72c4` 修正了续训中 warmup 设置被覆盖的问题。旧小样本第 258–410 步存在短暂学习率偏差；这些停止步数不能用于严格比较结构效率。

## 每轮判断与完整闭环评测

图闭环入口为 `src.evaluate_graph`，积木为 `src.evaluate_blocks`；读取对应 manifest、adapter checkpoint 与原始地图，保存实际轨迹和回答。逐轮关系、动作合法性、终止判断与整题到达／路径长度分别报告。

- `src.summarize_graph_eval` 能用真实图与 Q/V 重放现有图闭环日志，核对动作及后继，统计全部非终止轮、首轮和后续轮的关系指标。
- `src.evaluate_graph_readout` 和 `src.evaluate_kv_pilot` 的现有固定状态读数仅覆盖首轮。历史[首轮读出合同](configs/path_readout_gate.json)只约束这一子集，不再把通过它作为分析其他轮次或闭环结果的前置条件。
- **尚需补齐：**在完整示范历史下自由生成各轮回答，以及闭环按具体轮次、剩余距离、候选数量的分解；两者分开统计。具体缺口见[分析报告](results/readout_failure_analysis.md#4-每一轮都要看现有证据缺在哪里)。
- 候选整体重编号时同步更新动作、后继、Q 与答案。地图最近候选和环境最短动作分别评分；单候选轮选对不能等同于多候选比较成功。

## 暂不推进的草案

工作树中已有“自然目标、只从 Q 读取状态”的独立读出实现与两份未提交配置：`path_natural_qonly_ce.json`、`path_natural_qonly_rankids15.json`。没有 GPU 训练结果。

它沿用冻结地图、冻结 LLM 和校准 K/V；为 1024 道状态题各选一个使最近动作改变的真实训练目标，共 2048 道训练题，同图新目标验证／新图测试各 256 道。文字只列候选编号，Q 提供当前、目标和候选；动作与 Q 一起重编号。回答仍为完整排序、解释、动作，拟比较普通 CE 与仅排序编号 token ×15，各十轮。

这是改变文字信息条件的独立状态诊断，仍未覆盖完整轨迹各轮。草案保留，退出当前执行 TODO；是否需要它，先由有效任务的逐轮证据决定。

## 已归档的实验

小样本完整／短式 swap、扩大短式、十轮完整首轮及衍生诊断统一见[归档](results/archive/swap_training.md)。完整寻路中的候选 Q swap 与文字图、真实后继不一致；这条实验链不再用于选择下一步，不继续训练或等待剩余评测。原代码、配置和紧凑证据保留供追溯，不删除运行产物。

最小实现检查入口：

```bash
python -m unittest discover -s experiments/flamingo_map_reader/tests -p 'test_*.py'
```
