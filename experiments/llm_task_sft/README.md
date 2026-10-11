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
原始基座与 SFT 使用相同提示、候选编号和生成上限。每轮评 validation，按到达率选模，
并只对基座与选定模型评 test。reference 与自身历史 rollout 分开汇总。

## 实现与验证

- `protocol.py`：最优分支采样、短句标签、提示和控制标记解析。
- `prepare.py`：固定轨迹配额、token 数据、真实长度与输入曝光审计。
- `training.py` / `train.py`：全参数更新；FP32 主参数与 Adam 状态、BF16 运算；只在监督位置计算分块词表损失。
- `evaluate.py`：真实状态执行、停止判断与独立说明诊断；保留原始生成。
- 每个 epoch 保存完整 HF 模型、优化器、调度器及四份 RNG，全部写完后发布 `evaluation_ready.json`。
- `progress.json` / `train.jsonl` 记录实测步时、显存和 ETA；完成标记写入 `complete.json`。

本地 `python -m pytest experiments/llm_task_sft/tests -q` 验证轨迹认证、控制计分、
分块加权损失与全词表损失／梯度等价、真实 Qwen 对话标签。尚无研究结论。
