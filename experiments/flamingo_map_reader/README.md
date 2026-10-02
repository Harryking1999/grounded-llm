# 地图读取接口：寻路与积木

两项任务共用 Qwen2.5-1.5B-Instruct 的基座型号、地图记忆编码与门控 cross-attention 的结构和代码、整题 SFT 和逐轮文字协议。**寻路与积木分别新建并训练 P、cross-attention 等接口参数，保存到不同 checkpoint；不混合训练，也不共享训练后的接口权重。**地图与环境规则各自提供。设计与解释见 [DESIGN](DESIGN.md)，进度见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

讨论与汇报用的[失败分析报告](results/readout_failure_analysis.md)汇总同题小样本、K/V 解耦、尺度校准和短式 swap 监督的设置、数值、真实输出与改进建议；[紧凑证据](results/readout_failure_evidence.json)保留 loss 曲线及对应运行记录。短式模型已学会格式，但两题训练集仍均为 0/16 对原题／swap 同时正确，尚未通过地图读出。

当前 K/V 解耦合同：[寻路](configs/pilot_path256_addressed_kv.json)、[积木](configs/pilot_blocks_addressed_kv.json)。旧[寻路](configs/pilot_path256.json)、[积木](configs/pilot_blocks.json)配置保留为混合记忆基线。`src/text.py`、`src/sft.py` 统一英文开头、排序和终止回答；`src/train.py` 根据合同加载图 Q/V 或共享棋盘 Q。训练使用普通 Trainer batch、assistant-only CE，保持每轮 token 到当轮地图的方案 A 绑定。两个任务的训练超参数相同；实际 batch 由各自长题的显存测量决定。

新接口将角色与编号放在 attention key 路径，将完整 Q 经任务各自的投影 P 放在 value 路径，不加入关系模块 R。两题共用实现，但必须分别准备与配置完全一致的 manifest，并从头训练独立接口。其余文字、题集及训练合同沿用相应原配置。先用相同的小样本预算检查训练内排序、交换 Q 后的关系响应，以及同步重编号的一致性，再决定是否开展完整训练。

[成对小样本合同](configs/kv_pilot.json)从已冻结的完整 manifest 抽取寻路四图各若干题、积木互不重叠的棋盘题，分别生成旧混合记忆与 K/V 解耦的同题配置和 manifest。`src/prepare_kv_pilot.py` 是这一选择的唯一入口。小样本用于检测接口能否拟合关系和响应地图反事实，不替代正式的跨图／未见棋盘评测。
`src/evaluate_kv_pilot.py` 对两题的固定首步给出训练／验证的排序与最近候选分数；少量训练题额外保持文字和编号不动交换最近／最远候选 Q，并同步重排文字编号、地图槽和实际动作。报告交换后的新最近候选命中及重编号后是否仍选同一实际动作；全候选等距时不定义最近／最远交换。

首轮成对试点的结果见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。后续尺度检查由 `src/measure_value_channel.py` 在训练题真正预测排序变量 token 的位置测量：每层未乘 gate 的地图残差／hidden、实际残差／hidden。`src/prepare_kv_scale_followup.py` 依据[正式选择规则](configs/kv_scale_followup.json)与两题各自的测量报告，为每题固定一个全状态共享的 value 缩放常数，并生成同预算的普通整题与独立首轮反事实读出合同。反事实条件通过 `training.supervision_mode=counterfactual_first_turn` 选择，使用相同文字和编号的原 Q／交换 Q 成对样本；它是接口诊断，不是环境轨迹。

`src/evaluate_kv_pilot.py` 还可报告首个排序候选及 Q 交换前后成对正确率；`--scaffold-prefix` 补一个没有答案的公共回答开头，仅用于格式诊断，输出标记为 `scaffold_diagnostic`。积木另用 `src/evaluate_short_readout.py` 测两候选远近与最近候选短回答，包括同文字 Q 交换对照；这些结果与原完整排序分列。只有训练内关系读出与交换响应改善后才扩大训练覆盖。

## 数据与运行入口

以下路径由部署环境提供；所有 `runs/` 产物位于 Git 外。生成正式数据前，需让命令与已提交合同一致。

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

续训在同一训练命令上添加 `--resume .../models/checkpoint-N`。图闭环评测入口为 `src.evaluate_graph`，积木为 `src.evaluate_blocks`；两者都读取对应 manifest、adapter checkpoint 与原始地图。结果保存逐题实际动作和模型回答及紧凑 `results/summary.json`。图评测使用保留的原五图题集，积木按三组未见状态统计。

为排除小样本训练预算不足，[收敛对照合同](configs/kv_convergence.json)规定了现有十组条件的续训与统一评测。添加 `--convergence-contract` 并使用新的输出目录，从原 checkpoint 恢复接口、优化器、调度器和随机状态，保留原数据与 warmup。`convergence.jsonl` 定期记录同一训练集重新前向的整体 CE 和排序／短式选择 CE；两者持续进入平台才记为收敛，预算用尽不算收敛。按训练集整体 CE 保存 `models/best_loss`，不使用验证成绩挑权重。排序 CE 不包含末尾复制的 action，但仍是给定正确前文的条件损失，须另做自由生成和 Q 交换评测。

寻路另有独立的首步地图关系评测，使用[读出门槛](configs/path_readout_gate.json)。它在 64 条验证题和五图 120 条保留题的固定首状态生成一次回答，从冻结 Q/V 重新计算 `current` 与每个候选的远近，报告排序格式、逐对关系、完整排序、最近候选，以及动作是否遵循排序。验证题沿用训练示范的首轮文本和候选编号；保留题沿用闭环评测的首轮文本和编号。固定首步生成有单独的 token 上限，超过预算仍未给出关系和动作即计为失败。两个集合分别满足门槛，才称当前权重通过首步距离关系读出；这仍不代表整题规划成功。保留题闭环轨迹也可由 `src.summarize_graph_eval` 重放并按首轮／后续轮统计同样指标，用来定位后续状态的错误。

```bash
python -m experiments.flamingo_map_reader.src.evaluate_graph_readout \
  --config experiments/flamingo_map_reader/configs/pilot_path256_addressed_kv.json \
  --gate-config experiments/flamingo_map_reader/configs/path_readout_gate.json \
  --manifest runs/flamingo_map_reader/data/path_manifest.json \
  --source-root GRAPH_SOURCE_ROOT --adapter-checkpoint PATH_ADAPTER_PT \
  --out runs/flamingo_map_reader/path_readout
```

读出分数是对模型文字的检验；如果分数高，还需交换候选 Q 向量而保持文字不变，检查选择是否随地图变化，才能支持“使用了地图”这一因果解释。
同一入口加 `--splits train` 可对全部训练题首步作诊断；训练题只用于判断是否拟合，不参与上述验证与保留题门槛。

针对原权重读出失败，另备[关系 token 加权寻路配置](configs/pilot_path256_relation_weighted.json)作受控消融。它保留原整题数据、模型结构和文本格式，只在 assistant CE 中提高排序式的可变部分及 `<action>` 编号的权重；固定句式仍按普通权重计算。该配置须单独 `prepare` manifest、从新初始化训练并通过相同读出门槛评测，不能与原 checkpoint 续接或混为一次运行。代码同时适用于积木，但现有积木训练仍按已提交的原合同执行。

[固定 gate 寻路配置](configs/pilot_path256_fixed_gate.json)只将 cross-attention gate 固定为 `tanh(g)=0.1`，保持原寻路的普通整题 SFT、地图与训练超参数。它使投影和 cross-attention 从第一步就收到梯度，用于检验零 gate 初始化的优化障碍。独立准备 manifest 和训练 checkpoint，按训练、验证、保留三组的固定首步关系评测，再决定闭环评测。固定值是受控对照，不能单凭 gate 大小声称地图被使用。

最小正确性检查：

```bash
python -m unittest discover -s experiments/flamingo_map_reader/tests -p 'test_*.py'
```
