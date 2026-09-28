# 积木 Q-map：共享编码器与树关系监督

本目录只保留当前跨棋盘 Q 实验的训练、评测和报告入口。研究结论见[结果报告](results/report.md)，项目进度见[唯一状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。机器可读参数以 [configs/](configs/) 中的正式合同为准。

## 问题与边界

使用原论文 10×10 棋盘和八种简单形状，训练共享编码器 `Q(棋盘)`。离线精确求解器标注从一个状态到另一个状态的最短合法移除步数或已证明不可达；Q 只读取棋盘，目标在比较 `D(Q(后继), Q(目标))` 时才出现。评测时环境提供合法动作及真实后继，Q 负责选择；没有训练 V，也没有让 LLM 生成动作。

正式管线依次建立 1,000 张训练棋盘的多目标基础关系、同父分叉的树关系，再从已保存的树权重续训，补充清空目标与长程关系。训练、开发和封存测试按**整张初始棋盘**分开；封存测试只在开发集选定权重后运行。状态对分母不等于独立棋盘数。

## 代码接口

| 模块 | 职责 |
| --- | --- |
| `src/oracle.py`, `src/model.py` | 精确有向距离标签、合法后继、100 位棋盘编码器和固定 Q 距离 |
| `src/multiboard_data.py`, `src/multiboard_run.py` | 多棋盘多目标基础关系和共享 Q 训练 |
| `src/tree_supervision_data.py`, `src/tree_supervision_run.py` | 祖先、后代、同父完整分叉及换目标关系；在冻结数据上重训共享 Q |
| `src/tree_continue_data.py`, `src/tree_continue_run.py` | 保留既有关系和划分，补入清空与长程分叉并续训 |
| `src/multiboard_eval.py`, `src/rollout_eval.py` | 关系指标、逐步到达和首次不可达选择的共用函数 |
| `src/tree_supervision_eval.py`, `src/tree_breadth_eval.py` | 未见棋盘非空目标题、按起点／目标／最短距离分层的开发与封存题 |
| `src/tree_clear_eval.py`, `src/tree_length_eval.py`, `src/tree_step_eval.py`, `src/tree_distance_eval.py` | 完整初始棋盘清空、固定步数、每步选择和留出距离关系诊断 |
| `results/plot_tree_report.py` | 从冻结权重及保存的封存评测数据重绘报告图 |

数据和评测函数可由 Python 导入；训练与评测模块的 `main()` 提供命令行入口。任务池构造与权重评分分开：在开发集生成固定题后，可用同一题池比较多个 checkpoint；封存集使用另一批棋盘，并记录完整分母。精确 oracle 只用于离线标签或动作**选定后**的判卷，不进入 Q 的候选分数。

## 执行顺序

需要 `numpy`、`torch`、`h5py`，绘图另需 `matplotlib`。官方八形状数据集放在合同指定的忽略路径 `data/raw/`。从仓库根目录运行针对性检查：

```bash
python -m unittest discover -s experiments/blocks_distance_map/tests -v
```

三个正式训练入口及合同如下。合同中的父运行路径和源码身份指向**已经完成的历史运行**；重跑时须让父产物和合同匹配，不能把不匹配的 checkpoint 作为同一实验继续。

| 阶段 | 命令模块 | 合同 |
| --- | --- | --- |
| 多目标基础 Q | `experiments.blocks_distance_map.src.multiboard_run` | [multiboard_1000_multigoal.json](configs/multiboard_1000_multigoal.json) |
| 树关系 Q | `experiments.blocks_distance_map.src.tree_supervision_run` | [multiboard_1000_tree_supervision.json](configs/multiboard_1000_tree_supervision.json) |
| 清空与长程续训 | `experiments.blocks_distance_map.src.tree_continue_run` | [multiboard_1000_tree_continue.json](configs/multiboard_1000_tree_continue.json) |

例如，续训入口接收 `--config`、`--out`、`--source-commit` 和 `--device`；`--prepare-only` 与 `--train-only` 可分开准备数据和训练。其他训练入口的参数由其 `--help` 列出。正式运行使用已提交源码，输出独立写入 Git 忽略的 `runs/blocks_distance_map/`，保留 `config.json`、`source_commit.txt`、数据、检查点和摘要。

分层题先运行 `tree_breadth_eval` 的构题模式，指定 `--data`、`--config`、`--official`、`--split development|sealed` 和 `--out`；再用 `--tasks`、`--checkpoint`、`--out` 评分。其余评测模块提供独立 `--help`。本轮开发题、封存题、固定非空目标题、清空题和距离关系的运行文件名与权重身份见[报告末尾](results/report.md#评测边界与结果身份)。

旧的单棋盘、查表、困难分叉微调入口已从当前目录移除；相关历史可以从 Git 记录查看。
