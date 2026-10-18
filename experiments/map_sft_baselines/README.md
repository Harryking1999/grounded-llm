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

桥接凭据仅通过 stdin 进入进程内存，运行 profile、路径映射、PID、日志、缓存与权重全部在 Git 忽略的运行目录。训练端保留完整恢复状态，评测端只接收推理导出。评测 worker 异常仍记录并停止相应队列；桥接的连接、存储和传输错误会记录失败阶段、错误及下次重试时间，再关闭连接并重连。所有卡须空闲才领取任务，4090 的 0 号卡始终预留。

bridge 的 runtime profile 须包含 `config`（正式研究配置的绝对路径），以及源运行、目标运行、跳板/目标连接与 known_hosts 路径。停训策略从 runtime profile 读取；样本、评测规模和保存间隔从正式配置读取，不在队列中另写一套数值。

桥接每轮先下载汇总曲线和报告，再清理权重，最后发布新权重。重建评测端时按 step 合并训练端已有曲线，避免较短的新曲线覆盖历史数字。`baselines` 指定回传范围，`publish_baselines` 可单独指定发送权重的条件，避免与 028 独立评测重复。逐题记录不自动复制到训练端。`bridge_status.json` 实时记录连接、回传、清理、上传阶段；`bridge_failures.jsonl` 保留历次 I/O 失败记录，即使之后重连成功也不覆盖。

权重缓存策略由 [bridge_retention.json](configs/bridge_retention.json) 统一维护，profile 可用 `retention_config` 指定其绝对路径。默认只发布各条件最新 checkpoint，限制缓存总字节、各条件在途 checkpoint 数和每轮传输数；到达上限时保持回传结果并等待空间，不删除未完成评测来腾位置。用户账号配额与整盘剩余空间须分别检查。

自动清理仅删除推理 `weights.pt` 及中断上传的 `.part`，不删除评测、报告、状态或训练端存档。完整权重须满足：028 有同身份非空存档、所有配置模式评测完成、汇总数字已回传，且该 checkpoint 没有 worker 占位。已退役条件可在确认存档及无占位后清理。清理与领取任务共用调度锁；保留 `weights_retired.json` 和就绪元数据，防止重新领取或反复上传。中断且未发布就绪标记的上传会先清理再重试。

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

028 的训练卡释放后加入同一评测池。已有训练调度器时，评测入口加 `--training-queue TRAIN_RUN/training_queue.json`：有待分配训练时不领评测，卡号与模式读取正式配置的 `after_training_evaluation`。独立训练进程则用 `--after-training TRAIN_RUN/training/BASELINE/finished.json --gpus GPU_LIST`，只在成功完成最终导出后启动该组评测，失败／提前停止不放行；仍检查实际 GPU 显存空闲。配合 `--coordination-profile` 从 4090 共用池领取 RO／TF／probe，避免重复分片。等待期间不保持 SSH 会话；凭据经 stdin 传入。评测运行目录与训练目录分开，推理权重可通过只读链接复用。

## 验证与结果

## 10,000 条数据基线

用户授权的小数据对照使用 [b2_10k.json](configs/b2_10k.json) 和 [qwen7b_ra_10k.json](configs/qwen7b_ra_10k.json)。从已验收的 50,000 条物理轨迹中，按来源棋盘及真实最短长度分层抽取，共用同一批 10,000 条轨迹；每张棋盘十条，初始即目标保持原比例。抽样不新增棋盘、不改变动作、排序标签、地图或测试题。实际目标大小和长度分布保存在数据目录的 `distribution.json`，近似匹配不意味着每一联合分布完全相同。

B2 保持原 1.5B 地图全参接口与排序优先监督；7B 使用 Qwen2.5-7B-Instruct，无地图、全参、现有简短 Reason＋动作。用户确认暂不生成长 CoT，报告中应称为 7B R＋A 基线。7B 按自身 tokenizer 重新编码，输入物理轨迹相同。学习率、token 平均 CE、global batch、优化器和训练轮数沿用原合同；保存间隔按半轮随数据量缩放。7B 使用四卡 FSDP 分片，B2 使用两卡 DDP，不能把两者差异单独解释为地图因果收益。

028 的 7B 仅保留最新一份完整恢复状态；完整评测回传后，推理权重仅保留最新与最高完整到达率检查点。未完成评测的权重不删除。启用时在桥接 profile 中设置 `source_retention: latest_and_best`。4090 只持有受限推理缓存，评测后先回传数字再删除缓存；每组独立目录、配置、队列与指标，不能同 50,000 条旧曲线拼接。

会议讨论使用 [多基线讨论汇报](results/discussion_report.md)：集中说明数据、监督、基线、当前成绩与解释，不包含运行日志。

```bash
python -m unittest experiments.map_sft_baselines.tests.test_contract experiments.map_sft_baselines.tests.test_shard_pool -v
```

[results/report.md](results/report.md)记录准备、启动和验收；每条基线的半轮结果表独立维护。原始逐题回答和动态队列状态留在远端 `runs/map_sft_baselines/`，全局 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)只保留决策和入口。

## B2_10k 首步诊断

入口为 `src/decision_audit.py`，正式合同为 [b2_10k_decision_audit.json](configs/b2_10k_decision_audit.json)。固定 64 条非终止训练轨迹（来源棋盘互异）与 64 条非终止测试轨迹，在 1、4、10、20 epoch 自主生成首步；训练文字输入直接取原监督缓存，并核对与正式提示重建是否一致；地图由相同冻结 Q 重算。报告环境最短动作率、参考动作精确匹配率、首个排序候选的地图最优率、合法率及回答重复率，不以平均 CE 替代决策指标。

20 epoch 对全部 128 题追加文字替换、地图替换、地图向量清零。替换来源取自训练轨迹中的合法决策状态，与原题具有相同合法候选数、不同状态；文字替换保留原地图，地图替换保留原文字，清零保留槽位角色、地址和有效掩码。这些刻意不匹配的输入只用于检测敏感性，不作为能力提升或常规准确率。另在没有排序／理由泄露答案的固定 `<action>` 前缀后，计算同一参考动作完整数字序列的条件对数概率；这属于强制前缀探针，不是正常生成概率或归一化动作分布。各条件保存逐题配对输出，按 train／test 分开汇总。

同一固定题目的等更新步数对照使用 [b2_matched_update_control.json](configs/b2_matched_update_control.json)。完整科学结果见[专项诊断](results/b2_10k_diagnosis.md)，不在 README 记录运行日志。


## B2 小数据四组受控诊断

用户授权在 028 八卡、四个两卡通道开展学习率对齐、覆盖、重编号、MLP 四组；4090 的 1–4 号卡独立评测，0 号卡预留。正式合同为 `configs/b2_lr_match.json`、`b2_coverage.json`、`b2_renumber.json`、`b2_mlp.json`。四组从同一基座重训，Q 冻结，保持全参、排序优先监督；共同更新预算与逐步学习率由配置中的 `maximum_updates`、`scheduler_total_steps`、`warmup_steps` 明确分开。用户后续明确要求覆盖组也跑满 20 epoch，因此覆盖组延长至 150k 更新，其他三组仍为 50k。前 50k 更新保留为等预算比较，覆盖组最终结果另作同 epoch、不同曝光预算的比较。续训从原 50k 完整恢复断点接续，不重置优化器、RNG 或学习率；`src/extend_run.py` 等待原阶段成功结束后启动，`--extend-training` 只允许在固定显式学习率时程下延长 epochs 和更新上限。

覆盖选择入口 `src/coverage_subset.py` 在来源棋盘及最短长度配额内最大化新增非终止状态—目标对。预先采用的覆盖判据是原 50k 唯一对的 80%，只是执行用启发式，不是已证明的充分性阈值；10k 未达标时按用户授权改用 30k，不继续扩展。实际源数据有 338,664 个唯一对；原 10k 为 68,256，覆盖优选 10k 为 68,326，仅增加 70；30k 为 204,478（60.38%），仍未达到上述启发式阈值，但已经是用户授权的上限。原数据决策重复率很低，覆盖组应解释为增加不同经验为主的对照，不能把收益独归于筛选算法。详细分布和选样报告放生成数据目录，避免把唯一对计数误当成关系机制的证明。

MLP 复用 `AddressedMapMemoryEncoder.feature_ffn`：投影后的内容 `x` 变成 `x + Linear(GELU(Linear(x)))`，地址、角色与候选 ID 分支不变，所有状态共享参数。容量增加可能改善特征表达，也可能增加记忆；不能预先宣称减少文字依赖。

评测沿用固定题集 TF、RO、地图反转 probe，并在 probe 首轮增加地图清零、候选倒序重编号及候选数相同的文字替换。倒序重编号同步文字、地址及向量，按物理动作判一致。文字替换保留原地图、改变整个文字题面，属于刻意不匹配输入，只解释敏感性；缺少匹配 donor 时报告实际分母，不虚填准确率。另给四组共同训练交集中的 64 道首步题自由生成，用于与测试单步表现对照；该小样本不代替整题成绩。全部干预保存逐题回答与明确分母。旧固定评测集已参与决策，本轮定位原因，不将峰值当成独立泛化证据。

训练端通过既有 bridge 返回完整汇总后，删除评测端推理权重，保留逐题输出、摘要和清理标记。028 保留最新恢复状态及最新／最佳已评推理权重；未评完的权重不删除。采用每个配置独立的运行根与队列，不改写旧结果。源码以提交归档部署；启动身份、PID、输出路径和暂停的旧队列均保存到 Git 外 runtime 文件。低频观察训练吞吐与错误，禁止自动重启失败训练。
