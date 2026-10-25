# 新任务 Qmap：完整目标实验与固定目标试验

五类任务均可使用完整目标训练入口，原手电过桥与24点的固定目标入口保留复现。普通中间目标与多起终点关系的实施设计见[Qmap方法](../../tasks/QMAP_METHOD.md#任务扩展与训练安排)。各任务独立训练，复用公共Q编码器、距离、排序与采样。地图能力与证据从[总报告](results/report.md)进入，扩任务安排见[项目状态](../../docs/PROJECT_STATUS_AND_TODO.md)。

## 按任务进入

| 任务 | 规则与数据 | 配置 | 结果与限制 |
|---|---|---|---|
| 推箱子 | [任务入口](../../tasks/sokoban/README.md) | [配置目录](configs/sokoban/)、[固定分布收敛检查](configs/sokoban/sokoban_mixture_convergence.json) | [推箱子报告](results/sokoban/report.md) |
| 移车出库 | [任务入口](../../tasks/rush_hour/README.md) | [配置目录](configs/rush_hour/)、[卷积方案](configs/rush_hour/rush_hour_spatial_pairs.json) | [移车出库报告](results/rush_hour/report.md) |
| 棋子交换 | [任务入口](../../tasks/token_swapping/README.md) | [配置目录](configs/token_swapping/)、[近距离与排序方案](configs/token_swapping/token_swapping_local_walk8_rank4_pairs.json) | [棋子交换报告](results/token_swapping/report.md) |
| 手电过桥 | [任务入口](../../tasks/bridge/README.md) | [五人多目标](configs/bridge/bridge_multigoal_coverage.json)、[独立验收](configs/bridge/bridge_frozen_acceptance.json) | [带权目标与采用范围](results/bridge/report.md) |
| 24点 | [任务入口](../../tasks/game24/README.md) | [多目标](configs/game24/game24_multigoal_fit.json)、[查询关系留出](configs/game24/game24_goal_relation_holdout.json) | [运算层与原起点缺口](results/game24/report.md) |

多目标开发条件使用同一 `src.run` 的 `exact_state_pairs` 模式，配置预算均有限定；这些条件的完成不代表地图已通过正式验收。

| 任务 | 首批开发对照 | 新增信息与评测 |
|---|---|---|
| 推箱子 | [原关系](configs/sokoban/sokoban_multigoal_control.json)、[直接动作边](configs/sokoban/sokoban_multigoal_edges.json) | 同一小图、多目标、模型、预算和完整候选；教师路径上的Q下降、自主完成与首次循环／不可达步骤 |
| 移车出库 | [身份编码](configs/rush_hour/rush_hour_multigoal_identity.json)、[几何编码](configs/rush_hour/rush_hour_multigoal_geometry.json) | 同题同CNN；普通目标包含A未到出口的完整排列；几何编码保持场景内固定身份和A标记 |
| 棋子交换 | [原树长程](configs/token_swapping/token_swapping_multigoal_long.json)、[另一棵10顶点树](configs/token_swapping/token_swapping_alternate_tree.json) | 精确距离层、完整候选、动作边及训练两步环回放；新树独立训练，LLM读取另行验收 |
| 过桥 | [五人多目标](configs/bridge/bridge_multigoal_fit.json) | 全人员位掩码和手电位置图；Dijkstra精确耗时；训练、选择、总成本均包含即时耗时 |
| 24点 | [四数与中间目标](configs/game24/game24_multigoal_fit.json) | 每题全部运算DAG、各层精确多重集目标、可达与不可达候选；按约分有理数跨起题合并曝光 |

过桥与24点复用任务层 `relation_data.py` 和 `qmap.py`。过桥人数与24点槽位数保存在checkpoint范围中，不同人数／槽位拒绝错配。24点裁判可指定完整中间目标，默认四数到24的旧调用保持可用；五数动作与特征槽位已支持，尚不因此宣称五数地图或LLM已训练。开发集与验证集用于选模；最终验收和权重采用仍按原流程执行。

三个棋盘／图任务的验收保留全部合法动作，不搜索或重试；各必需组完整到达率至少80%，目标90%以上，见[验收要求](configs/exact_pair_acceptance.json)。过桥另以已冻结的新耗时场景检查成本最优性，五人通过、六人未通过；24点尚未采用。旧固定目标试验仍由各自pilot配置维护。环境提供真实后继，地图学习距离与候选排序。

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

采样记录默认同步压缩。可用单个有界后台线程压缩原样数组；完整断点写入前必须等所有记录成功落盘，写入异常会阻止断点替换。过桥和24点的实际百步记录分别约405MB、614MB，单次压缩耗时约13秒、12秒，是已测CPU等待来源。同步／后台对照见[同步配置](configs/bridge/bridge6_trace_sync.json)和[后台配置](configs/bridge/bridge6_trace_async.json)；实现检查要求权重、随机状态、曝光和记录数组一致。循环吞吐排除了断点等待，因此性能结论同时检查包含写入等待的总耗时。

实际匹配检查中，后台压缩把包含断点等待和最终评测的阶段耗时从109.92秒降到73.62秒，点关系循环吞吐约14.87万→25.58万/秒；显存峰值相同。最终权重逐位一致，随机状态、曝光和70个记录数组也一致。该结果仅说明记录写入的性能改善，不表示模型能力提高，见[结果](results/shared/trace_write_performance.json)。

| 能力 | 模块入口 |
|---|---|
| 统一命令入口与模式分流 | [run.py](src/run.py) |
| 完整状态对训练、数据、评测、诊断 | [relations.py](src/relations.py)、[relation_data.py](src/relation_data.py)、[relation_evaluate.py](src/relation_evaluate.py)、[relation_diagnose.py](src/relation_diagnose.py) |
| 早期固定目标试验，保留复现入口 | [data.py](src/data.py)、[evaluate.py](src/evaluate.py)、[diagnose.py](src/diagnose.py) |

评测与诊断的参数见对应模块`--help`。任务规则和适配在[任务实现](../../grounded_llm/tasks/)，公共方法见[Qmap说明](../../tasks/QMAP_METHOD.md)。早期试验与当前三个任务条件不同，配置和结果分别在[pilot配置](configs/pilot/)和[pilot结果](results/pilot/)，不纳入当前地图验收。
