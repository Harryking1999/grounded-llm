# 项目状态与有序 TODO

本页维护研究优先级、各任务结论与待办。完整数值比较见[任务能力表](../tasks/README.md#任务能力总表)，模型身份见各任务 selection，运行状态需读取对应 runtime；这里不报告实时进程。

## 当前主线

当前主线为积木、二阶魔方、8-Puzzle与寻路的四任务联合SFT，比较地图全参与3B、7B文字全参；另行检验清空训练后的非空目标OOD，并补1.5B文字对照。汉诺塔暂不处理，过桥与推箱子待规则和地图就绪后再评跨任务迁移。匹配提示、题集与预算，以完整任务成功为主。研究定义与证伪标准见[研究简述](RESEARCH_BRIEF.md)。

采用已确定的五任务配置，不再为难住 GPT 追加难度。Qwen3-32B 不能可靠完成是当前任务筛选条件，GPT 作为强基线。Qmap 继续采用已有 MLP、距离 Huber 与原权重 soft-rank；改进优先考察数据覆盖、关系组合与采样，分别核对可查询覆盖、实际训练曝光和真实求解能力。

## 各任务结论

| 任务 | 当前结论 | 依据 |
|---|---|---|
| 积木 | 正式范围为12块清空；采用图889/1000，尚未达到900门槛；历史510题单列回归，非空目标仅辅助诊断 | [采用模型](../tasks/blocks/Qmap/selection.json)、[报告](../experiments/blocks_distance_map/results/blocks12_joint_report.md) |
| 寻路 | 采用统一方法 Q 及已选权重，旧 Q/V 仅作历史对照 | [采用模型](../tasks/path/Qmap/selection.json)、[报告](../experiments/path_qmap/results/report.md) |
| 汉诺塔 | 已采用八盘地图，正式范围为任意合法起点到指定柱整塔；不追加训练 | [采用模型](../tasks/hanoi/Qmap/selection.json)、[报告](../experiments/hanoi_qmap/results/report.md) |
| 二阶魔方 | 采用指定复原断点；保留二阶条件，三阶探索已停止，复原与关系泛化仍有限 | [采用模型](../tasks/cube2/Qmap/selection.json)、[报告](../experiments/cube2_qmap/results/report.md) |
| 8-Puzzle | 采用98.28%复原权重；中间目标与严格关系保持仍不足，已有窗口收尾后不追加训练 | [采用模型](../tasks/puzzle8/Qmap/selection.json)、[报告](../experiments/puzzle8_qmap/results/report.md) |

五任务纯文字强基线及补充对照已完成，见[统一报告](../experiments/task_text_baselines/results/report.md)。旧积木 SFT 与这些条件不同，不能直接相减归因。

## 短期待办

1. **推进四任务联合SFT及匹配基线。**积木、二阶魔方、8-Puzzle、寻路各12,500条；地图1.5B两卡、文字3B两卡、文字7B四卡，global batch均为4，超参数沿用最近三组。平台补1.5B文字全参的12块清空对照；4090用原三组实际停止的最新完整断点评测非空目标OOD，冻结图本身单列参照。原三组已停止并保留完整恢复状态，5万步比较与地图OOD参照见[结果报告](../experiments/map_sft_baselines/results/report.md#5-12块清空三组50k数据的阶段结果)；没有10万步成绩。配置及运行入口见[实验入口](../experiments/map_sft_baselines/README.md)。
2. **维护任务文字与裁判，落实逐轮地图／CoT接口。**任务原文与[集中副本](../TASK_TEXTS.md)保持同步。
3. **优先改进积木清空地图。**保留当前图及完整恢复状态，以固定1000题和历史510题共同验收；已完成的采样对照见报告，不机械追加同配方训练。其他任务保持所选权重。
4. **保留数据量诊断结论。**当前采用50k：原50k与100k同10万步分别为466/510、469/510，30k加训仍明显不足；旧诊断不再追加。此结论不证明普适样本门槛或唯一机制；冻结Q曝光审计暂缓，见[诊断比较](../experiments/map_sft_baselines/results/b2_four_controls.md)。

## 后续研究

- 任务内OOD与多任务训练分别核对地图和接口的训练信息范围；跨任务迁移只在对应规则、地图和未见题集准备后评测。
- 手电过桥、推箱子、24点已完成首轮固定目标Qmap可行性试验：过桥成本表示较有希望但未稳定最优；推箱子未见布局泛化不足；24点主要检验可解性，第一步仍易失败。[Flash64k固定筛查与加难探索](../experiments/task_text_baselines/results/report.md)中，Boxoban hard原题362耗尽64k且最终答案为空；24点加难21题全部完成，保留结果并停止加难。原过桥8至128人21题全部最优；256至4096人的15题全部完成、仅2题最优，长人数作为计划压力诊断单列。三者尚无采用图，不进入多任务SFT；地图限制见[报告](../experiments/new_tasks_qmap/results/report.md)。
- Rush Hour与Token Swapping的[Flash64k筛查](../experiments/task_text_baselines/results/report.md)已完成，各40题，分别17/40、36/40，成功答案均最优。25题思考耗尽，Rush Hour另2题误报无解；完成、最优成本和预算失败单列。保留的新地图方案为：[Boxoban hard推箱子](../tasks/sokoban/Qmap/README.md)10×10四箱、[Rush Hour](../tasks/rush_hour/Qmap/README.md)无墙6×6，各由同一任务Qmap覆盖多布局、输入完整棋盘、检验未见布局；每题指定一个完整成功目标以对齐单目标读取，训练与评测只用指定目标，旧Flash成绩不改称指定目标表现。[Token Swapping](../tasks/token_swapping/Qmap/README.md)先固定公开10顶点树学习任意状态对距离，验收后再考虑12–15顶点。[指定目标15题筛查](../experiments/task_text_baselines/results/exact_goals_flash64k.json)已完成：推箱子0/2、Rush Hour0/8、固定10顶点树4/5且成功均最优；11份失败均为64k思考耗尽，结果只描述本批已知起点及等价重标题。精确图、独立状态对训练与冻结reader入口已实现，配置见[正式实验](../experiments/new_tasks_qmap/README.md)；先完成广覆盖地图，必需组贪心至少80%、目标90%以上，再考虑LLM。尚无这些条件的采用权重，不按文字筛查登记采用。其余过桥／渡河[候选](../tasks/README.md#新任务候选难度与地图关系)保留调查结论。
- 在贪心之外比较采样与 pass@k 时，明确总预算、独立采样与整题裁判口径。

入口：[任务工作区](../tasks/README.md)、[Qmap 方法](../tasks/QMAP_METHOD.md)、[实验实现](../experiments/README.md)、[开发机手册](../DEVELOPMENT_MACHINES.md)。文献与旧会议不维护当前计划。
