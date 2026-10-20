# 地图与 SFT：同数据多基线

本实验在共同物理轨迹与 token CE 下比较地图信息、主干训练范围和接口容量。研究设计见 [DESIGN.md](DESIGN.md)，核心成绩见[主报告](results/report.md)；固定510题已经用于模型选择，不能把最高值当成独立泛化成绩。

## 条件与结果入口

| 比较 | 正式配置 | 结果与解释 |
|---|---|---|
| 50k 主比较 | [study.json](configs/study.json) | [主报告](results/report.md) |
| B1 恒定学习率与逐轮重编号 | [设计 §7.3](DESIGN.md#73-b1-改进恒定学习率与逐轮重编号) | [主报告](results/report.md) |
| 10k 地图全参与7B无地图 SFT | [b2_10k.json](configs/b2_10k.json)、[qwen7b_ra_10k.json](configs/qwen7b_ra_10k.json) | [主报告](results/report.md)、[首步诊断](results/b2_10k_diagnosis.md) |
| 学习率、覆盖、重编号、内容 MLP 四组诊断 | [b2_lr_match.json](configs/b2_lr_match.json)、[b2_coverage.json](configs/b2_coverage.json)、[b2_renumber.json](configs/b2_renumber.json)、[b2_mlp.json](configs/b2_mlp.json) | [同预算比较](results/b2_four_controls.md) |
| 100k 嵌套数据对照 | [b2_100k.json](configs/b2_100k.json) | [诊断报告中的证据边界](results/b2_four_controls.md) |
| 原50k B2 同更新预算补评 | [b2_50k_evaluation.json](configs/b2_50k_evaluation.json) | [诊断比较](results/b2_four_controls.md)；逐题结果在已有补评 runtime |
| 50k 内容 MLP：有／无逐轮重编号 | [b2_50k_mlp.json](configs/b2_50k_mlp.json) | 共用原50k积木轨迹、冻结图与510题 |
| 二阶魔方、8-Puzzle 的50k地图 SFT | [cube2](configs/cube2_50k_mlp_renumber.json)、[puzzle8](configs/puzzle8_50k_mlp_renumber.json) | 各任务采用图，MLP＋逐轮重编号；固定目标复原 |
| 缓存与权重保留 | [bridge_retention.json](configs/bridge_retention.json) | 具体运行状态在各 runtime |

地图组现行监督为“排序 → Reason → Action”；旧 reason-first 条件单列历史。7B 使用简短 Reason＋Action，不能称为长 CoT。不同数据量、主干大小或训练预算的结果分别解释；不拼接曲线。

覆盖组的同更新比较与更长训练后的最终点分开。100k 原样保留50k数据，再按 DESIGN 的长度、覆盖和多样性顺序扩充；不加入重编号或 MLP。SFT 的起终点未重合不等于冻结 Q 或选模过程无泄露。

## 代码归属

| 内容 | 实现 |
|---|---|
| 数据准备与监督顺序 | [prepare.py](src/prepare.py)、[积木 SFT 适配](../../grounded_llm/tasks/blocks/sft.py)、[重编号数据集](../../grounded_llm/tasks/blocks/dataset.py) |
| 覆盖选样、嵌套扩充与曝光审计 | [coverage_subset.py](src/coverage_subset.py)、[extend_data.py](src/extend_data.py)、[leakage_audit.py](src/leakage_audit.py) |
| 参数解释、reader 与训练 | [公共 SFT](../../grounded_llm/core/sft)、[reader](../../grounded_llm/core/reader)、[实验训练入口](src/train.py) |
| 参考状态、自主 rollout 与内容干预 | [evaluate.py](src/evaluate.py)、[metrics.py](src/metrics.py)、[decision_audit.py](src/decision_audit.py) |
| 固定题集地图贪心参照 | [map_policy_reference.py](src/map_policy_reference.py) |
| 训练及评测队列 | [queue.py](src/queue.py)、[evaluation_queue.py](src/evaluation_queue.py)、[shard_pool.py](src/shard_pool.py) |
| 跨机传输与结果回传 | [bridge.py](src/bridge.py) |
| 已有训练的预算延长 | [extend_run.py](src/extend_run.py)；需对应配置与完整恢复状态 |

规则、提示、数据变换与单步裁判位于 `grounded_llm/tasks/blocks/`；reader、地图槽位、排序评分与 SFT 训练位于 `grounded_llm/core/`。本目录选择配置、准备数据并编排运行；`common.py`、`modeling.py` 等旧路径只作兼容转发。修改任务规则进入任务层，修改公共训练进入 core，不复制一套训练／评测脚本。

## 运行入口

以下为接口形式，不表示许可重启或扩大已有运行；实际配置、机器与预算以本次任务为准。

```bash
python -m experiments.map_sft_baselines.src.prepare --config CONFIG --sources SOURCES --model-path MODEL --out DATA --workers WORKERS
python -m experiments.map_sft_baselines.src.queue train --config CONFIG --data DATA --model-path MODEL --root RUN
python -m experiments.map_sft_baselines.src.queue evaluate --config CONFIG --data EVAL_DATA --model-path MODEL --q-checkpoint Q --root EVAL_RUN
python -m experiments.map_sft_baselines.src.bridge --profile PRIVATE_RUNTIME_PROFILE
```

runtime profile 保存路径、节点、known_hosts 与传输映射；凭据仅通过 stdin 进入内存。模型与机器入口见[开发机手册](../../DEVELOPMENT_MACHINES.md)。正式参数从配置读取，不在队列或说明中另抄。

二阶魔方与8-Puzzle用 `src.restoration_prepare` 准备数据，训练、评测分片、桥接和缓存清理仍用上述入口。任务文字、物理动作、重编号和裁判位于 [restoration_sft.py](../../grounded_llm/tasks/restoration_sft.py)，复用已有任务规则和提示原文。魔方沿用QTM六动作与复原目标；8-Puzzle沿用所选地图的标准目标 `1 2 3 / 4 5 6 / 7 8 0`。

新任务各取50,000个不同非终止起点，不单独加入初始即终止样本；完整轨迹保留到达后的停止监督。先从全部可达状态锁定统一随机留出题，再用冻结Q选动作，只保留每步真实最短距离减少一且最终复原的路线。按最短路长度递减选择，同长度优先增加未见状态—目标覆盖；任何训练轮次均排除留出题初始状态。固定目标、有限状态空间下不沿用积木的绝对覆盖门槛；真实覆盖、长度分布、筛选比例和最长token数保存在数据 `acceptance.json`。题集不按Q成败筛选；Q本身的训练曝光不因此被排除。

四组从同一基座初始化，沿用原B2的优化器、学习率时程、batch、训练预算、生成和动作预算。内容MLP沿用既有诊断实现；重编号每个epoch重绑候选ID，不新增物理样本。028每组两卡；4090使用1–4号卡，评完且结果回传后删除推理缓存。028保留最新完整恢复状态及已评最佳模型；未评权重保留等待队列。

## 多节点评测

多节点评测通过同一主节点的 `.worker.claim/` 原子占位；非共享磁盘的节点通过 SFTP 访问这个目录。已有 worker 用 `--resume-workers` 接管，不能因等待过久就释放占位或重复领取。辅助节点使用独立 `--node-id` 和 `--coordination-profile`，完整答卷回传后发布摘要。训练后评测入口已有 `--training-queue` 或 `--after-training`，无需另造监控脚本。

bridge 先回传结果、再按配置清理缓存、最后发布推理权重。训练端保留完整恢复状态；评测未完成或仍有占位的权重不能清理。账号配额与整盘空间分别检查，跨机原始答卷、缓存状态和失败日志留在 runtime；传输重试不构成重启训练的授权。

## 结果维护

[主报告](results/report.md)保留关键条件和能力比较；[四组诊断](results/b2_four_controls.md)保留原因定位；[讨论稿](results/discussion_report.md)用于已有汇报语境。完整曲线、逐题回答、曝光统计、PID 和运行时间线留在 Git 外。报告没有更新不代表进程仍在运行。
