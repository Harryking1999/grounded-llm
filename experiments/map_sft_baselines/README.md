# 地图与 SFT：同数据多基线

本实验在全部 1,000 张来源棋盘上增强地图贪心成功且最短的示范，统一数据与 token CE，对照地图信息、可训练主干与接口。研究假设和监督示例见 [DESIGN.md](DESIGN.md)，正式参数只维护在 [configs/study.json](configs/study.json)。

## 代码入口

| 模块 | 职责 |
|---|---|
| `src/prepare.py` | 全来源采样、长度优先选样、最短认证、两种输出缓存与覆盖验收 |
| `src/common.py` | 共同状态、固定编号、R＋A／R＋排序＋A 与伪地图 |
| `src/contract.py` | 统一解释样本数、batch、保存间隔、评测规模与平台期规则 |
| `src/metrics.py` | 不依赖模型/GPU 的已保存回答汇总 |
| `src/modeling.py` | 复用既有 reader，配置主干训练范围，标准 token CE |
| `src/train.py` | 两卡 DDP、梯度累积、全局 token 归一化及半轮断点 |
| `src/evaluate.py` | 共同参考状态 TF、自主 RO、既有地图 NDCG 与内容探针 |
| `src/queue.py` | 四训练通道、空闲卡接后续基线、最新检查点优先评测与整点汇总 |
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

## 验证与结果

```bash
python -m unittest experiments.map_sft_baselines.tests.test_contract -v
```

[results/report.md](results/report.md)记录准备、启动和验收；每条基线的半轮结果表独立维护。原始逐题回答和动态队列状态留在远端 `runs/map_sft_baselines/`，全局 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)只保留决策和入口。
