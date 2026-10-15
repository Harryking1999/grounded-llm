# 地图与 SFT：同数据多基线

本实验在全部 1,000 张来源棋盘上增强地图贪心成功且最短的示范，统一数据与 token CE，对照地图信息、可训练主干与接口。研究假设和监督示例见 [DESIGN.md](DESIGN.md)，正式参数只维护在 [configs/study.json](configs/study.json)。

地图组按用户决策重新从基座训练：`B1_sortfirst`、`B2_sortfirst` 替代原 B1/B2，输出改为“排序 → reason → action”。物理任务、地图向量、动作与 reason 内容不变，只调整输出顺序及对应提示。原 B1/B2 保留为停止的 reason-first 对照，不再领取新评测；B0/B3 继续原训练。新旧运行使用不同目录和曲线，不能拼接 epoch。

用户另行授权的 B1 改进运行为 `B1_const_renumber`：恢复旧恒定学习率配方，并逐轮重排编号，仍冻结主干、使用共同增强全集和排序在前监督。它使用独立目录，见 [设计第 7.3 节](DESIGN.md#73-b1-改进恒定学习率与逐轮重编号)；不改变五条主基线的训练合同。

## 代码入口

| 模块 | 职责 |
|---|---|
| `src/prepare.py` | 全来源采样、长度优先选样、最短认证、两种输出缓存与覆盖验收 |
| `src/reorder_supervision.py` | 复用已验收路线与地图缓存，仅重排地图监督并重新编码 |
| `src/common.py` | 共同状态、固定编号、R＋A／R＋排序＋A 与伪地图 |
| `src/contract.py` | 统一解释样本数、batch、保存间隔、评测规模与平台期规则 |
| `src/metrics.py` | 不依赖模型/GPU 的已保存回答汇总 |
| `src/map_policy_reference.py` | 在固定 510 题上比较地图 Top-1 与 Top-3 均匀随机，多种子并与完整 B2 结果配对 |
| `src/modeling.py` | 复用既有 reader，配置主干训练范围，标准 token CE |
| `src/train.py` | 两卡 DDP、梯度累积、全局 token 归一化及半轮断点 |
| `src/evaluate.py` | 共同参考状态 TF、自主 RO、既有地图 NDCG 与内容探针 |
| `src/queue.py` | 训练通道和统一评测入口、最新检查点优先及整点汇总 |
| `src/evaluation_queue.py` | 多节点评测 worker 启动、在途 worker 接管、统一结果发布 |
| `src/shard_pool.py` | 主节点原子占位、所有权核验、按需权重传输与完成标记 |
| `src/bridge.py` | 只传推理权重、完整发布后入队、紧凑指标回传及平台期标记 |

复用 `blocks_distance_map` 的规则与求解器、`flamingo_map_reader` 的接口/排序计分、`llm_task_sft` 的事实依据和控制解析；不复制旧模型或改写旧运行。

## 运行方式

```bash
python -m experiments.map_sft_baselines.src.prepare --config experiments/map_sft_baselines/configs/study.json --sources SOURCES --model-path MODEL --out DATA --workers 24
python -m experiments.map_sft_baselines.src.queue train --config experiments/map_sft_baselines/configs/study.json --data DATA --model-path MODEL --root RUN
python -m experiments.map_sft_baselines.src.queue evaluate --config experiments/map_sft_baselines/configs/study.json --data EVAL_DATA --model-path MODEL --q-checkpoint Q --root EVAL_RUN
python -m experiments.map_sft_baselines.src.bridge --profile PRIVATE_RUNTIME_PROFILE
```

桥接凭据仅通过 stdin 进入进程内存，运行 profile、路径映射、PID、日志、缓存与权重全部在 Git 忽略的运行目录。训练端保留完整恢复状态，评测端只接收推理导出。失败记录后停止相应队列，不静默重试。所有卡须空闲才领取任务，4090 的 0 号卡始终预留。

bridge 的 runtime profile 须包含 `config`（正式研究配置的绝对路径），以及源运行、目标运行、跳板/目标连接与 known_hosts 路径。停训策略从 runtime profile 读取；样本、评测规模和保存间隔从正式配置读取，不在队列中另写一套数值。

## 多节点评测

4090 保存统一 `models/`、`evaluation/` 和结果总表；辅助节点只保存自己的模型缓存和回答。节点不共享磁盘时，辅助队列通过 SFTP 访问同一主节点目录，不能各自在本地加锁后当作跨节点互斥。

1. 分片唯一键为基线、checkpoint、模式及题目区间。所有队列在主节点原子创建对应 `.worker.claim/`，成功后才启动 worker；`owner.json` 保存节点、PID 与随机所有权 token。
2. 辅助节点按需下载推理权重，先回传完整 `cases.jsonl`，最后发布 `summary.json`；发布后释放自己的占位。完成标记存在即跳过，只有主节点写完整汇总。
3. 锁不按时间自动过期。队列异常时保留占位，防止慢 worker 被重复领取；显式 `--resume-workers` 会核对本节点实际进程，接管活 worker，或回传已完成的退出 worker。确认本节点 worker 已退出且没有完整结果后才释放其旧占位。
4. 每台机器另有队列锁，防止重复启动调度器。启用新节点前先接管现有 worker 的占位，不能让无锁的旧调度器继续领取任务。

```bash
# 主节点：与现有在途 worker 接续，不重跑已完成回答
python -m experiments.map_sft_baselines.src.queue evaluate --config CONFIG --data DATA --model-path MODEL --q-checkpoint Q --root RUN --node-id 4090 --resume-workers
# 辅助节点：SSH 凭据仅经 stdin 送入内存
python -m experiments.map_sft_baselines.src.queue evaluate --config CONFIG --data DATA --model-path MODEL --q-checkpoint Q --root LOCAL_RUN --node-id platform-35016 --gpus 0,1 --coordination-profile RUNTIME_PROFILE --resume-workers
```

辅助 runtime profile 复用桥接的公开 SSH 字段与 `target_run`（统一主节点目录），不保存凭据。未来节点采用相同入口、不同 `node-id` 和本机 GPU 配置。各节点动态队列状态、占位、PID 与日志留在远端运行目录。

028 的训练卡释放后加入同一评测池，优先补 TF 与地图内容探针，其他节点继续覆盖 RO。评测入口加 `--training-queue TRAIN_RUN/training_queue.json`：有待分配训练时不领评测，B4 优先接空出的训练通道；之后只使用训练队列已释放且显存空闲的卡。具体卡号和评测模式读取正式配置的 `after_training_evaluation`，不停止或占用现有训练。评测运行目录与训练目录分开，推理权重可通过只读链接复用。

## 验证与结果

```bash
python -m unittest experiments.map_sft_baselines.tests.test_contract experiments.map_sft_baselines.tests.test_shard_pool -v
```

[results/report.md](results/report.md)记录准备、启动和验收；每条基线的半轮结果表独立维护。原始逐题回答和动态队列状态留在远端 `runs/map_sft_baselines/`，全局 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)只保留决策和入口。
