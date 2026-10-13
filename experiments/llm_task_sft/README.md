# 无地图积木全参数 SFT

检验在相同棋盘、规则与工具更新下，任务 SFT 能否改善原始 1.5B 的解题能力。
模型每轮读取当前状态、历史和全部合法候选，生成一行可验证依据和一个动作／停止标记。
离线求解器只制作最短轨迹标签和评分，不进入模型推理。

正式参数以 [blocks_reason_bs4.json](configs/blocks_reason_bs4.json) 为准；
数据组成与比较边界见 [方案](../../docs/LLM_SFT_PLAN.md)。数据、训练输出与逐题回答放在忽略的运行目录。

## 运行接口

从包含 `experiments/` 的代码根目录运行；路径由运行时传入。

```bash
python -m experiments.llm_task_sft.src.prepare --config CONFIG --source-manifest SOURCE --model-path BASE --out DATA
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc-per-node=4 -m experiments.llm_task_sft.src.train --config CONFIG --data DATA --model-path BASE --out TRAIN
CUDA_VISIBLE_DEVICES=0 python -m experiments.llm_task_sft.src.evaluate --config CONFIG --data DATA --model-path MODEL --out SHARD --split validation --mode rollout --shard-index 0
```

先用 `prepare --smoke-tasks 32` 检查真实标签，再用 `train --smoke-steps 20 --longest-smoke`
在最长样本上完成四卡前后向、优化器更新和完整保存；正式训练不使用 smoke 数据。
原始基座与 SFT 使用相同提示、候选编号和生成上限。前三轮原方案评 validation 并补齐与地图线相同的 510 题，
历史回答与选模记录保留。用户随后授权继续训练，并指定后续仅评这 510 题，详见下节。

## 半轮续训与监测

正式续训合同为 [blocks_reason_continuation.json](configs/blocks_reason_continuation.json)。
从第三轮完整断点恢复模型、Adam、调度器与各卡 RNG；沿用固定 50,000 条数据和 global batch 4。
原三轮 cosine 已到末尾，后续保持其终值学习率 2e-6，不重新 warmup 或改变原三轮权重。

```bash
python -u -m experiments.llm_task_sft.src.continuation --profile PROFILE
```

同一台 4090 的四卡依次执行：训练 6,250 步 → 保存完整断点 → 完成 510 题的 rollout 与 reference → 再训练。
因此新评测点为 3.5、4、4.5…… epoch，不补训已过去的半整数点。半轮恢复接续同一 epoch 的剩余样本，
完整模型和所有训练状态均恢复；执行信息记录在运行目录，不使用 1,000 题 validation。

不设 epoch 上限，也不因连续几次分数不涨自动停止。每个半轮保留模型、恢复状态、逐题回答与曲线；
`curve.json` 的峰值仅是固定 510 题监测集上的最佳观察值，不能当作独立 test 泛化成绩。
进程失败或磁盘不足时停止并记录原因，不自动更改配方或删除旧产物。
运行目录的 `stop_after_cycle` 可要求完成当前训练与评测周期后停止。

## 实现与验证

- `protocol.py`：最优分支采样、短句标签、提示和控制标记解析。
- `prepare.py`：固定轨迹配额、token 数据、真实长度与输入曝光审计。
- `training.py` / `train.py`：全参数更新；FP32 主参数与 Adam 状态、BF16 运算；只在监督位置计算分块词表损失。
- `evaluate.py`：真实状态执行、停止判断与独立说明诊断；保留原始生成。
- `bridge.py` / `epoch_queue.py`：前三轮的模型转移与分片评测；续训由 `continuation.py` 串行协调训练、510 题评测和曲线汇总。当前机器与卡号见续训合同。
- 迁移可复用已验收的完整 split 汇总，跳过该 split 的生成；来源记录在运行目录中。平台逐题回答仍留在原共享盘，未取回前不能在 4090 查询这些历史回答。
- 每个 epoch 保存完整 HF 模型、优化器、调度器及四份 RNG，全部写完后发布 `evaluation_ready.json`。
- `progress.json` / `train.jsonl` 记录实测步时、显存和 ETA；完成标记写入 `complete.json`。

本地 `python -m pytest experiments/llm_task_sft/tests -q` 验证轨迹认证、控制计分、
分块加权损失与全词表损失／梯度等价、真实 Qwen 对话标签。

阶段验收已核对训练数据配额、前两轮完整断点及全部已完成评测分片。
三轮 validation 普通到达率为 3.2%、11.6%、25.7%，初始已达标题均正确停止 100/100；
多数普通题仍最终无合法动作，依据模板的事实正确性也未稳定。
三轮同题 510 评测已全部完成，到达率为 3.53%、11.76%、27.25%。
续训 3.5、4、4.5 点为 31.76%、32.16%、31.57%，出现短期平台迹象；loss 仍下降，尚未确认上限。
已验收六个完整断点、累计 252 个分片的动作回放与汇总，当前自动接续至第 5 轮，见[验收报告](results/acceptance.md)。
用户随后授权从第三轮继续训练并逐半轮评测；35016 已登记，当前串行续训与评测均在 4090。
进度与精确计数见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

## 合法候选随机对照

`random_legal.py` 在固定题目上精确计算三种随机策略的到达概率：
全部几何合法候选、额外排除破坏目标格的候选、保护目标且优先大块后随机。三者均假定正确识别目标、没有格式错误或提前停止；
不筛选可达动作，不搜索后选路，不进行多次尝试择优。初始已达标题单独统计。
动作只会移除格子，因此可通过动态规划累加全部成功路线的概率；失败分支仍保留在选择分母中。
同时报告每题首轮，以及固定参考轨迹上全部非终止状态的随机可达动作率和最短进展率；
每个状态先计算候选比例，再对状态取平均，不能把单步概率直接当成整题到达率。

```bash
python -m experiments.llm_task_sft.src.random_legal --config experiments/llm_task_sft/configs/blocks_random_legal.json --data DATA --out DIAGNOSTIC
```

同题首轮、单步与整题对照结果见[合法候选分析](results/legal_candidate_baselines.md)。
510 test 使用 `blocks_random_legal_test.json`；loss 下降速度、监督与参数差异见[优化诊断](results/optimization_diagnostics.md)。

## 保存回答的策略比较

`analyze_policies` 在同一 510 题上比较已保存的 SFT 与地图回答，不生成新回答。
先按真实动作回放棋盘，再定位首次不可逆失败：误删目标格、保住目标却留下不可拼除的残余，
以及可解状态上的提前停止、非法动作或格式错误。死局之后的回答不重复计为首次失败。
按决策轮报告仍可解题数、首次失败、累计到达，并比较成功题集合和物理动作；候选局部编号不作为动作身份。

```bash
python -m experiments.llm_task_sft.src.analyze_policies --config CONFIG --sft-run SFT_RUN --map-run MAP_RUN --map-cases SAVED_MAP_ROOT --out OUTPUT
```

正式范围见 [blocks_policy_comparison.json](configs/blocks_policy_comparison.json)。
当前地图的内容依赖另用 [配对探针合同](../flamingo_map_reader/configs/blocks_current_map_content_probe.json)：
同一状态重新生成正常与反序地图两种回答，历史、文字、候选动作和生成预算保持一致，
实际动作与可达性评分和排序评分分开；不将这项小样本探针当作完整 rollout 成绩。
