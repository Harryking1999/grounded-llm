# 三个新任务的 Qmap 实验

推箱子、移车出库、棋子交换分别训练地图，每题指定一个完整目标；先验收地图能否独立贪心完成，再考虑接入LLM。三个任务均未达标，成绩与证据从[总报告](results/report.md)进入。研究优先级见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

## 按任务进入

| 任务 | 规则与数据 | 配置 | 结果与限制 |
|---|---|---|---|
| 推箱子 | [任务入口](../../tasks/sokoban/README.md) | [配置目录](configs/sokoban/)、[当前多轮训练](configs/sokoban/sokoban_coverage_epochs_pairs.json) | [推箱子报告](results/sokoban/report.md) |
| 移车出库 | [任务入口](../../tasks/rush_hour/README.md) | [配置目录](configs/rush_hour/)、[卷积方案](configs/rush_hour/rush_hour_spatial_pairs.json) | [移车出库报告](results/rush_hour/report.md) |
| 棋子交换 | [任务入口](../../tasks/token_swapping/README.md) | [配置目录](configs/token_swapping/)、[近距离与排序方案](configs/token_swapping/token_swapping_local_walk8_rank4_pairs.json) | [棋子交换报告](results/token_swapping/report.md) |

验收保留全部合法动作，不搜索、不重试、不自动排除死路。环境提供真实后继，地图学习距离与候选排序，不代表学会动作转移。各必需组完整到达率至少80%，目标90%以上，见[共同验收要求](configs/exact_pair_acceptance.json)。

## 目录与文件职责

```text
configs/             共同验收要求、固定布局名单
  sokoban/           推箱子训练、数据构造与诊断配置
  rush_hour/         移车出库配置
  token_swapping/    棋子交换配置
  pilot/             早期手电过桥、小棋盘推箱子和24点配置
results/
  report.md          三任务同口径比较与证据入口
  sokoban/           推箱子报告与结果
  rush_hour/         移车出库报告与结果
  token_swapping/    棋子交换报告与结果
  shared/            跨任务数据、输入变化与评测检查
  pilot/             早期试验结果
src/                 运行、数据、训练、评测和诊断入口
tests/               规则、标签、留出边界、采样及恢复检查
```

配置与结果沿用原文件名；JSON内容保留原样，结果内的提交、命令和旧路径仍是历史来源。根目录的[布局名单](configs/sokoban_coverage_layouts.json)被多份已运行配置直接引用，因此保留原路径；新目录中的原配置仍可与已有完整断点匹配。配置存在不代表已完成；Softsign为未运行候选，原扁平回放条件已停止，状态见推箱子报告。

原始数据、日志、PID、采样记录与checkpoint放在忽略的`runs/new_tasks_qmap/`等运行目录；采用权重只维护在`tasks/<task>/Qmap/selection.json`。本地目录整理不迁移远端运行目录或已提交源码快照。

## 推箱子覆盖量对照

| 条件 | 配置入口 | 要检验什么 |
|---|---|---|
| 布局覆盖 | [64张](configs/sokoban/sokoban_coverage64_pairs.json)、[1,000张](configs/sokoban/sokoban_coverage1000_pairs.json) | 同题、同预算比较布局扩展 |
| 相邻关系 | [原采样](configs/sokoban/sokoban_neighborhood_control_pairs.json)、[相邻成组](configs/sokoban/sokoban_neighborhood_coupled_pairs.json) | 同目标的当前状态与后继监督 |
| 完整局部路径 | [原采样](configs/sokoban/sokoban_local_path_control_pairs.json)、[打散](configs/sokoban/sokoban_local_path_shuffled_pairs.json)、[成组](configs/sokoban/sokoban_local_path_joint_pairs.json) | 新路径数据及同批成组的作用 |
| 关系覆盖与多轮训练 | [当前配置](configs/sokoban/sokoban_coverage_epochs_pairs.json) | 逐条遍历准备关系、降学习率、固定已训练样本与区间平均loss |
| 训练失败与路径构造 | [失败诊断](configs/sokoban/sokoban_training_failure_diagnosis.json)、[路径数据](configs/sokoban/sokoban_local_paths_data.json)、[固定验证](configs/sokoban/sokoban_neighborhood_validation.json) | 训练允许范围内的循环、分支、死锁和连续路径 |

比较和限制见[推箱子报告](results/sokoban/report.md)。多轮条件从原采样路径对照的验证所选权重开始，新建优化器与采样阶段，不作为完整续训；epoch只指当前准备关系池的实际遍历，不代表穷尽物理状态。预算和学习率只维护在配置中。完整断点保存优化器、关系顺序和随机状态；实时状态见忽略的`coverage_epochs_runtime_40048.json`，旧路径三组见`local_path_runtime_40048.json`。

## 运行入口

从仓库根目录执行。以下为新建棋子交换基线的准备与训练入口；其他条件替换配置和输出目录，配置内的初始化或数据引用仍相对输出目录解释。

```bash
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --prepare-only
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --device cuda:0 --resume
```

正式运行使用已提交源码，远端操作前按[机器手册](../../DEVELOPMENT_MACHINES.md)核对进程和资源。恢复使用原配置及完整`last.pt`；结果JSON或推理权重不能替代恢复状态。

| 能力 | 模块入口 |
|---|---|
| 统一命令入口与模式分流 | [run.py](src/run.py) |
| 完整状态对训练、数据、评测、诊断 | [relations.py](src/relations.py)、[relation_data.py](src/relation_data.py)、[relation_evaluate.py](src/relation_evaluate.py)、[relation_diagnose.py](src/relation_diagnose.py) |
| 早期固定目标试验，保留复现入口 | [data.py](src/data.py)、[evaluate.py](src/evaluate.py)、[diagnose.py](src/diagnose.py) |

评测与诊断的参数见对应模块`--help`。任务规则和适配在[任务实现](../../grounded_llm/tasks/)，公共方法见[Qmap说明](../../tasks/QMAP_METHOD.md)。早期试验与当前三个任务条件不同，配置和结果分别在[pilot配置](configs/pilot/)和[pilot结果](results/pilot/)，不纳入当前地图验收。
