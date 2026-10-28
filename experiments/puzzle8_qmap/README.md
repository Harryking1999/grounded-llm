# 数字华容道 Qmap

研究共享状态编码器Q能否表达状态关系并支持连续求解。规则为3×3棋盘，空格沿上下左右移动，每次一步；标准复原态所在分量共有181440个状态。

## 当前优化问题

采用图的标准复原率为98.28%，中间目标仍弱。当前优先处理实际编号1–999目标族和中央空格目标，同时保持复原及旧关系。已知棋盘对称与数字重命名可将目标归为角、边、中央三类，其读取结果与绝对状态编码分别比较。数值和机制见[结果报告](results/report.md)，采用身份见[selection](../../tasks/puzzle8/Qmap/selection.json)。

## 实现入口

| 内容 | 入口 |
|---|---|
| 状态图、特征和目标坐标变换 | [environment.py](../../grounded_llm/tasks/puzzle8/environment.py) |
| 原共享MLP和冻结读取 | [model.py](../../grounded_llm/tasks/puzzle8/model.py) |
| 独立训练与恢复 | [run.py](src/run.py) |
| 全图关系、结构与遍历采样 | [graph_data.py](../../grounded_llm/tasks/puzzle8/graph_data.py) |
| 当前失败、原成功轨迹与目标分层 | [failure_sampling.py](src/failure_sampling.py)、[目标泛化配置](configs/goal_generalization.json) |
| 原固定中间目标及完整策略核验 | [checkpoint_audit.py](src/checkpoint_audit.py)、[progress_report.py](src/progress_report.py) |

训练复用原L1距离Huber与完整合法候选soft-rank；状态特征及共享MLP保持原形式。原关系、复原、广域关系和失败后续共同训练，当前分布补足实际目标族，并按循环影响的起点数提高重点分叉的采样权重。原成功轨迹持续重放。

沿用已有完整距离表、7077604个结构关系和固定监测题；模型、Adam、广域遍历位置及随机状态从采用图的完整断点接续。实际目标、距离层和重点分叉曝光进入原运行记录，结合训练误差、复原、三个参照目标的全状态执行及原中间目标判断训练效果。

训练和恢复使用 `python -m experiments.puzzle8_qmap.src.run`。准备数据用 `--prepared-run`，完整标签表用 `--full-distance-table`；多卡执行通过 `--device`／`--devices` 分摊同一全局批量。中断后用同一配置、输出目录和 `--resume-run` 接续。

原模型选择与正式验收条件见对应配置。完整断点、采样状态、逐题结果和运行位置保存在Git外的runtime；历史比较集中于结构化结果。
