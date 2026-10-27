# 二阶魔方 Qmap

研究共享状态编码器 Q 能否用固定向量距离表达状态关系，并支持连续求解。规则为 U、U′、R、R′、F、F′ 六动作，每次四分之一转计一步；固定 DBL 角块参考系，共3,674,160个状态，最远距离14步。

## 当前优化问题

优先改善非复原目标及同起点换目标，同时保留复原能力。FLAN 的非复原目标来自编号1–999，实际距复原态6–13步；旧的1–3步近目标结果单列。当前成绩、机制诊断与证据见[结果报告](results/report.md)，采用地图见[selection](../../tasks/cube2/Qmap/selection.json)。

## 实现入口

| 内容 | 入口 |
|---|---|
| 任务规则、精确距离和状态特征 | [environment.py](../../grounded_llm/tasks/cube2/environment.py) |
| 共享MLP、轴对称适配与冻结读取 | [model.py](../../grounded_llm/tasks/cube2/model.py) |
| 当前目标泛化训练 | [near_goal_continuation.py](src/near_goal_continuation.py)、[配置](configs/goal_generalization.json) |
| 全状态、多目标关系与原关系重放 | [full_state_joint.py](src/full_state_joint.py) |
| 当前策略循环、出口与正确关系保持 | [cycle_repair.py](src/cycle_repair.py) |
| 覆盖数据与固定池对照 | [coverage_data.py](src/coverage_data.py)、[coverage_train.py](src/coverage_train.py) |

监督使用精确距离Huber和完整六候选soft-rank。环境提供合法后继，Q分别编码状态，按固定距离贪心选动作。训练数据由广域起点、多目标、近目标入口、原复原路径关系和训练区错误决策组成；统计实际目标、距离层、起点与失败重放曝光，结合训练曲线及验证执行判断进展。

当前按已知状态上的关系补全评估。原复原留出关系及各评测父节点、全部候选到目标的关系，均按正反方向和六种轴对称呈现保护。原复原能力参与验证选择，最终另核对全状态复原。

完整状态、动作图和距离规则的数学分析见[结构说明](results/structure_notes.md)。历史配置对应各自原始运行，源码、Adam、采样状态和逐题轨迹保存在运行目录。
