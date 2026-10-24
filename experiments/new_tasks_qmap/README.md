# 新任务 Qmap：完整目标实验与固定目标试验

推箱子、移车出库、棋子交换使用完整目标训练入口；手电过桥与24点已有固定目标试验入口。各任务独立训练，复用公共Q编码器、距离、排序与采样。地图能力与证据从[总报告](results/report.md)进入，扩任务安排见[项目状态](../../docs/PROJECT_STATUS_AND_TODO.md)。

## 按任务进入

| 任务 | 规则与数据 | 配置 | 结果与限制 |
|---|---|---|---|
| 推箱子 | [任务入口](../../tasks/sokoban/README.md) | [配置目录](configs/sokoban/)、[固定分布收敛检查](configs/sokoban/sokoban_mixture_convergence.json) | [推箱子报告](results/sokoban/report.md) |
| 移车出库 | [任务入口](../../tasks/rush_hour/README.md) | [配置目录](configs/rush_hour/)、[卷积方案](configs/rush_hour/rush_hour_spatial_pairs.json) | [移车出库报告](results/rush_hour/report.md) |
| 棋子交换 | [任务入口](../../tasks/token_swapping/README.md) | [配置目录](configs/token_swapping/)、[近距离与排序方案](configs/token_swapping/token_swapping_local_walk8_rank4_pairs.json) | [棋子交换报告](results/token_swapping/report.md) |
| 手电过桥 | [任务入口](../../tasks/bridge/README.md) | [固定目标配置](configs/pilot/bridge.json) | [首轮结果](results/pilot/summary.json) |
| 24点 | [任务入口](../../tasks/game24/README.md) | [固定目标配置](configs/pilot/game24.json) | [首轮结果](results/pilot/summary.json) |

完整目标三任务的验收保留全部合法动作，不搜索或重试；各必需组完整到达率至少80%，目标90%以上，见[验收要求](configs/exact_pair_acceptance.json)。过桥和24点的首轮范围由各自pilot配置维护，环境提供真实后继，地图学习距离与候选排序。

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

配置与结果沿用原文件名；JSON内容保留原样，结果内的提交、命令和旧路径仍是历史来源。根目录的[布局名单](configs/sokoban_coverage_layouts.json)被多份已运行配置直接引用，因此保留原路径；新目录中的原配置仍可与已有完整断点匹配。Softsign为未运行候选，原扁平回放条件已停止，状态见推箱子报告。

原始数据、日志、PID、采样记录与checkpoint放在忽略的`runs/new_tasks_qmap/`等运行目录；采用权重只维护在`tasks/<task>/Qmap/selection.json`。本地目录整理不迁移远端运行目录或已提交源码快照。

## 推箱子覆盖量对照

| 条件 | 配置入口 | 要检验什么 |
|---|---|---|
| 布局覆盖 | [64张](configs/sokoban/sokoban_coverage64_pairs.json)、[1,000张](configs/sokoban/sokoban_coverage1000_pairs.json) | 同题、同预算比较布局扩展 |
| 相邻关系 | [原采样](configs/sokoban/sokoban_neighborhood_control_pairs.json)、[相邻成组](configs/sokoban/sokoban_neighborhood_coupled_pairs.json) | 同目标的当前状态与后继监督 |
| 完整局部路径 | [原采样](configs/sokoban/sokoban_local_path_control_pairs.json)、[打散](configs/sokoban/sokoban_local_path_shuffled_pairs.json)、[成组](configs/sokoban/sokoban_local_path_joint_pairs.json) | 新路径数据及同批成组的作用 |
| 关系覆盖与多轮训练（已停止） | [配置](configs/sokoban/sokoban_coverage_epochs_pairs.json) | 扩大曝光后跨布局误差仍变差，保留完整断点 |
| 完整求解结构小验证 | [配置](configs/sokoban/sokoban_solution_structure_pilot.json) | 匹配比较零散终局关系、完整路径与分支、路径内不同跨度的中间目标关系 |
| 分支连续覆盖与定向修复 | [分支](configs/sokoban/sokoban_branch_closure_pilot.json)、[优化](configs/sokoban/sokoban_branch_closure_refine.json)、[回放](configs/sokoban/sokoban_branch_failure_replay.json) | 补全替代最优后继及一步恢复，分别检查训练兼容性和未训练中间目标 |
| 完整目标可达区域 | [短预算](configs/sokoban/sokoban_graph_basins_pilot.json)、[匹配拟合预算](configs/sokoban/sokoban_graph_basins_12000.json) | A原分支、B完整允许可达区域、C多目标、D整目标图的训练执行反馈 |
| 连续图采样与分级反馈 | [配置](configs/sokoban/sokoban_connected_graph_sampling.json) | 固定C数据和匹配初始化，比较原采样、布局与目标均衡的连续邻域、分级整图反馈 |
| 原大分布与完整图混合 | [配置](configs/sokoban/sokoban_broad_graph_mixture.json) | 保留原1,000张布局，比较原关系、完整目标图及两种混合比例；共用初始化、更新数、损失与固定验证 |
| 固定混合分布收敛检查 | [配置](configs/sokoban/sokoban_mixture_convergence.json) | 接续现有20%混合组的完整优化状态，不变采样；检查全图拟合、训练分支执行和原留出泛化，保留节点及验证改善权重 |
| 固定分布延长候选（未运行） | [配置](configs/sokoban/sokoban_mixture_convergence_extension.json) | 仅延长同一混合组的完整优化阶段；额外预算待授权 |
| 训练失败与路径构造 | [失败诊断](configs/sokoban/sokoban_training_failure_diagnosis.json)、[路径数据](configs/sokoban/sokoban_local_paths_data.json)、[固定验证](configs/sokoban/sokoban_neighborhood_validation.json) | 训练允许范围内的循环、分支、死锁和连续路径 |

比较和限制见[推箱子报告](results/sokoban/report.md)。多轮条件从原采样路径对照的验证所选权重开始，新建优化器与采样阶段，epoch只指当前准备关系池的实际遍历，预算和学习率只维护在配置中。完整断点保存优化器、关系顺序和随机状态；实时状态见忽略的`coverage_epochs_runtime_40048.json`，旧路径三组见`local_path_runtime_40048.json`。

结构与采样诊断复用[训练器](src/relations.py)和[评测器](src/relation_evaluate.py)，各 `solution_*` 模块负责具体数据组织或诊断。结果与下一步依据归[推箱子报告](results/sokoban/report.md)，本页不重复维护阶段结论。

## 运行入口

从仓库根目录执行。以下为新建棋子交换基线的准备与训练入口；其他条件替换配置和输出目录，配置内的初始化或数据引用仍相对输出目录解释。

```bash
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --prepare-only
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --device cuda:0 --resume
```

正式运行使用已提交源码，远端操作前按[机器手册](../../DEVELOPMENT_MACHINES.md)核对进程和资源。恢复使用原配置及完整`last.pt`。

| 能力 | 模块入口 |
|---|---|
| 统一命令入口与模式分流 | [run.py](src/run.py) |
| 完整状态对训练、数据、评测、诊断 | [relations.py](src/relations.py)、[relation_data.py](src/relation_data.py)、[relation_evaluate.py](src/relation_evaluate.py)、[relation_diagnose.py](src/relation_diagnose.py) |
| 早期固定目标试验，保留复现入口 | [data.py](src/data.py)、[evaluate.py](src/evaluate.py)、[diagnose.py](src/diagnose.py) |

评测与诊断的参数见对应模块`--help`。任务规则和适配在[任务实现](../../grounded_llm/tasks/)，公共方法见[Qmap说明](../../tasks/QMAP_METHOD.md)。早期试验与当前三个任务条件不同，配置和结果分别在[pilot配置](configs/pilot/)和[pilot结果](results/pilot/)，不纳入当前地图验收。
