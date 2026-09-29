# 寻路外部地图接口

本实验检验：在同一张图上学到的 Q/V 距离，能否帮助 LLM 选择合法后继并到达目标。地图给出的候选分数是 `||Q(当前节点) + V(动作) − Q(目标)||₂`，不是精确最短步数。环境提供真实当前位置和合法边；模型选择动作；精确最短路只用于离线判卷。

## 当前五图协议

[正式合同](configs/path256_continuous_five_graphs.json)冻结五张 256 节点图、120 道题、每题 16 次重复和整题输出预算。四个条件如下：

| 条件 | 生成与反馈 |
| --- | --- |
| Instruct 无地图 | 一次生成整条路线，以最终 `{"path": [...]}` 提交 |
| Thinking 无地图 | 同样一次生成整条路线 |
| Instruct＋地图 | 每次用 `<action>节点</action>` 提交后继；控制器核验后，向同一回复追加实际状态、已确认路径、合法后继及 learned-map 距离 |
| Instruct＋地图，优先最小距离 | 与上一组相同，提示词另加“优先选择距离最小的合法后继” |

地图组在**第一个完整的新节点**处暂停，不等整条路线生成完。程序从起点开始维护已确认路径，拒绝不存在的边和重复节点；到达目标即结束。地图更新不包含真实最短距离。无地图组没有中途插入。四组的路径都按真实图重放；合法到达是主要指标，最短路、路径长度、非法动作、截断和 token 用量另列。因为地图组还有逐节点反馈，与无地图组的差异不能只归因于距离信息。

| 模块 | 职责 |
| --- | --- |
| `src/q_map.py`、`src/transitions.py` | 读取图 Q/V、给候选评分、执行并核验真实动作 |
| `src/evaluate_path256_continuous.py` | 提示词、动作边界、同一回复续写与答案提取 |
| `src/evaluate_path256_continuous_batch.py` | 按正式合同运行和断点跳过已有记录 |
| `src/summarize_path256_continuous_batch.py` | 独立重放记录并汇总成绩 |

运行产物保存在 Git 忽略的 `runs/`，正式参数只维护在 `configs/`。模块命令行参数可由 `python -m experiments.external_map_interface.src.evaluate_path256_continuous_batch --help` 和对应汇总模块的 `--help` 查看。当前进度见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)；协议案例、原始提示片段与阶段结果见[五图报告](results/path256_five_graphs.md#新实验设置连续生成过程中提供地图)。

同一批 120 题的无地图 Qwen3-32B Thinking 单次采样已完成验收：合法到达 92/120，最短路 61/120；与 4B Thinking 首次采样的逐题对照、失败类型和案例见[32B 验收报告](results/path256_qwen32b_thinking.md)。

## 已完成的旧条件

旧版五图逐步决策共 5,760 条，要求最短路且每一步重新发送图与状态，结果见[五图报告](results/path256_five_graphs.md)和[结果摘要](results/path256_five_graphs.json)。单图和独立图复测见[path256 距离报告](results/path256_distance.md)、[独立图报告](results/path256_diverse.md)。早期单棋盘积木 Q/V 与状态条件位移是转移拟合诊断，见[积木试验结果](results/blocks_q_pilot.md)；当前跨棋盘、只训练 Q 的积木研究在[独立目录](../blocks_distance_map/README.md)。
