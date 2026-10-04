# 地图读取：完整长轨迹 SFT

本实验检验：冻结 LLM 与地图，只训练地图读取接口，能否从完整轨迹学会远近排序、行动、停止和总结。当前主实验为单图寻路与 1000 初始棋盘积木；旧小样本用于历史诊断。寻路 epoch 1 的训练内逐轮诊断已取得 100% 控制格式／合法动作、69.2% 地图最优动作，尚不构成泛化结论。

## 阅读顺序

1. [训练与评测设置](DESIGN.md)：数据范围和数量 → 动作覆盖与编号 → 初始／更新文本 → Top-10 监督 → 训练与评测。
2. [结果报告](results/readout_failure_analysis.md)：先记录寻路 epoch 1，后续在同页补充。
3. [项目状态与有序 TODO](../../docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页。

正式合同为 [path_single_long.json](configs/path_single_long.json) 与 [blocks1000_long.json](configs/blocks1000_long.json)；[紧凑证据](results/long_trajectory_summary.json)保存实际数据统计、动作覆盖及结果分母，不另复制配置。

## 分支、源码与运行对应

| 对象 | 本轮对应关系 |
|---|---|
| 实现分支 | `codex/long-trajectory-training` |
| 训练协议提交 | `f73b700`；`087e2d3` 已将监督改为排序与 Top-10，`f73b700` 对应测试断言修正 |
| 后续分支提交 | 至 `f8161fb`，均为评测队列改动：`3992d6d`／`57d8ff1`／`d24e345`／`4ec7e78` 为重启、子进程认领与记录，`aec7166` 增加跨机器委派，`9f66c2b` 增加每 epoch 验证闭环电池，`f8161fb` 修正队列排序；均未改变本轮训练数据／监督代码 |
| 较早方案分支 | `codex/full-trajectory-protocol` 的 `1ae0e0e` 是运行前方案，仍含数值距离监督要求，不能作为当前实际设置 |
| 远端运行根目录 | `/zhanghanyue/experiment/flamingo_map_reader/runs/long_f73b700_20261004` |
| 原始训练源码 | `/zhanghanyue/experiment/flamingo_map_reader/code/long_f73b700`；后续队列可使用独立的新源码目录，运行名保持不变 |

以源码提交、运行合同和数据清单识别实验，不以当前窗口检出的分支或临时进程号判断设置。旧 `long_9234c5c_20261003`、原四图、16／4 小样本及 swap 均不并入本轮成绩。

运行根目录内，`path/` 与 `blocks/` 各自保存 `data/manifest.json`、`data/trajectories/*.pt`、`training/config.json`、`training/models/`。`evaluation/<task>/<checkpoint>/...` 保存评测分片的完整回答和摘要。清单给出划分与样本数，训练合同绑定地图、模型、配置和实际 batch；这些大型／运行产物不提交 Git。

## 代码职责

| 源码 | 职责 |
|---|---|
| [prepare_trajectories.py](src/prepare_trajectories.py) | 物理任务选择与划分、训练后缀排除、数据覆盖及地图贪心参照 |
| [prompt.py](src/prompt.py)、[blocks_prompt.py](src/blocks_prompt.py)、[text.py](src/text.py) | 初始规则、环境更新、终止与历史总结文本 |
| [sft.py](src/sft.py)、[blocks_sft.py](src/blocks_sft.py) | 地图贪心示范，排序式答案与 Top-10 选择 |
| [trajectory_protocol.py](src/trajectory_protocol.py)、[trajectory_dataset.py](src/trajectory_dataset.py) | 自洽重编号、固定版本缓存；不改变真实轨迹 |
| [transcript.py](src/transcript.py)、[memory.py](src/memory.py)、[fusion.py](src/fusion.py) | assistant-token 标签、逐轮地图绑定、K/V 分离与门控读取 |
| [train.py](src/train.py) | 冻结基座，完整轨迹普通交叉熵训练及接口权重保存 |
| [trajectory_eval.py](src/trajectory_eval.py)、[trajectory_metrics.py](src/trajectory_metrics.py)、[relations.py](src/relations.py) | reference／闭环自由生成、关系与控制评分 |
| [trajectory_queue.py](src/trajectory_queue.py)、[trajectory_smoke.py](src/trajectory_smoke.py) | 已有训练／评测调度与长样本显存检查 |

历史结果入口：[原完整轨迹与小样本](results/archive/full_trajectory_history.md)、[swap](results/archive/swap_training.md)。旧配置与诊断源码保留复现，不代表当前训练入口。
