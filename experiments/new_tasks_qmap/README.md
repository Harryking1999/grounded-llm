# 三任务完整状态对 Qmap

当前实验训练推箱子、Rush Hour、Token Swapping三张独立共享地图。每次查询只给一个完整目标，独立编码当前、目标与所有合法后继。复用core的StateEncoder、距离Huber、soft-rank与MapBatch；不改变已有任务权重，不训练LLM。

| 任务 | 正式配置 | 精确真值与覆盖 |
|---|---|---|
| Boxoban hard，10×10四箱，玩家也须到位 | [sokoban_pairs.json](configs/sokoban_pairs.json) | 对可枚举布局建立完整物理图，逐个完整成功目标计算有向距离；墙体旋转镜像整组留出 |
| Rush Hour，无墙6×6，所有车位置须匹配 | [rush_hour_pairs.json](configs/rush_hour_pairs.json) | 公开无墙题的完整可达分量、全部成功目标；静态车辆配置与等价重命名整组留出 |
| Token Swapping，固定公开10顶点树 | [token_swapping_pairs.json](configs/token_swapping_pairs.json) | 完整10!排列与归位精确BFS表；任意目标重标号仅作教师标签；端点及关系独立留出 |

[验收合同](configs/exact_pair_acceptance.json)要求各任务、各必需测试组的冻结地图贪心到达率至少80%，目标90%以上，原始困难起点另列。不能平均掉未见布局／目标的失败，也不能用搜索、oracle过滤或目标集合代替本条件。没有通过验收的模型不登记采用。

## 准备、训练与恢复

```bash
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --prepare-only
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --device cuda:0 --resume
```

另两任务替换配置与输出目录。正式运行使用已提交源码；启动前核对35016进程和空闲卡，只用授权且实际空闲的卡。输出目录拒绝覆盖；`--resume`只接受原配置，恢复优化器、随机数、采样顺序、曝光计数，不能拿推理权重冒充恢复。

`grounded_llm/tasks/<task>/relation_data.py`负责环境图和oracle，实验层`src/relation_data.py`负责留出与监督。训练遍历全部入选训练状态，混合距离层采样，并持续更换完整目标；实际梯度曝光和样本流分别保存。训练状态全集不等于状态对全集，Token Swapping不枚举约13万亿对关系。二值原始输入可无损缓存到GPU；缓存不含标签，也不预计算带oracle信息的Q。

保存`data/graphs/`、固定`relations.npz`、端点划分、实际训练样本分片、`progress.jsonl`、完整`best.pt`/`last.pt`、初始与最终评测、曝光及资源统计。达到完整训练源覆盖后才选正式候选最佳断点；此前断点只供恢复。验证按配置指标选权重，测试不参与选权重。

## 冻结评测与信息条件

`src/relation_evaluate.py`报告距离MAE与分层误差、最优候选top-k、方向性、Bellman一致性、不可达识别、贪心到达与最优率、超额成本、循环及具体案例。贪心只比较全部合法后继的`1 + distance(Q(child), Q(goal))`，到精确目标才完成。同一批查询比较oracle、简单任务启发式和仅即时成本。规划查询不重复抽样，并分配布局与距离层预算。

推箱子的合法死路仍保留。精确穷举图证明的不可达记为无穷；训练的有限代理单独标识，不声称有限向量等于无穷。资源界导致未完成枚举的场景只记unknown并报告排除原因，不能标作无解。Rush Hour和Token Swapping采用对称距离，跨不同物理场景不定义最短路。

三个任务的`grounded_llm/tasks/<task>/qmap.py:MapReader`直接加载冻结权重，接公共CURRENT/GOAL/CANDIDATES结构。环境提供真实后继，地图不因此被解释为学会状态转移。Flash指定目标结果见[15题固定诊断](../task_text_baselines/results/exact_goals_flash64k.json)，历史目标条件结果不混算。

验证：`python -m unittest discover -s experiments/new_tasks_qmap/tests -v`。包含小图独立BFS、任意目标等价、不可逆死锁、候选／关系泄漏、reader冻结与完整恢复检查。C++ oracle在Linux运行时额外验证。

# 历史固定目标可行性试验

检验手电过桥的加权成本、推箱子的不可逆死路、24点的算术可解性，能否被现有状态 MLP 和距离／排序监督表示。每次调用只处理一个任务；三个模型、数据和结果独立，不共用权重，也不改动已有任务训练器。

| 任务 | 规则／数据入口 | 配置 |
|---|---|---|
| 手电过桥 | [任务](../../tasks/bridge/README.md) | [bridge.json](configs/bridge.json) |
| 推箱子 | [任务](../../tasks/sokoban/README.md) | [sokoban.json](configs/sokoban.json) |
| 24点 | [任务](../../tasks/game24/README.md) | [game24.json](configs/game24.json) |

## 运行

```bash
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/bridge.json --out runs/new_tasks/bridge --source-commit COMMIT --device cuda:0
```

另两任务替换配置和输出目录。使用35016已确认空闲GPU，沿用平台项目Python；启动前复查占用。执行有限配置预算后退出，不追加实验。源码以已提交归档传输；远端使用节点临时盘运行，运行结束回存持久盘。具体路径与PID只写本地忽略的 `runs/new_tasks_qmap/runtime.json`。

输出包括原始图、来源题目、场景索引、初始模型指标、训练日志、best/last完整状态、逐状态预测和最终摘要。checkpoint存储优化器和采样状态，但本入口未实现恢复参数；不能把best权重称为已支持精确续训。已有目录拒绝覆盖。

## 比较与解释

编码器不单独拼接查询目标，但推箱子的输入通道含目标格，因此当前向量依赖目标布局。本轮只监督同场景既定目标的距离，不是任意目标地图验证。推箱子目标是集合，使用目标条件参照向量。不可达仍用真值正无穷保存，Huber与soft-rank使用配置中的有限代理值；不能声称有限向量表示了无穷距离。过桥按即时耗时＋后继距离评分，其余两任务即时成本为1。

每个任务报告训练场景、验证场景及留出场景的有限距离误差、最短动作正确率、严格候选排序、选入死路比例、整题成功与最优数。验证成功率及动作率选checkpoint；测试只作固定初始和最终读数，不参与选择。所有策略共享合法候选、起点和固定步数上限。oracle使用精确距离；简单启发式分别是尚未过桥者的最长耗时、最小箱子目标曼哈顿匹配、离24最近的当前数值；额外保留只看即时成本及未训练模型。没有搜索、死路屏蔽、反悔或额外推理预算。

数据公开源见各任务README；自行生成的过桥和小推箱子场景不宣称是标准benchmark。重复训练曝光保存在运行结果中；24点初题划分不保证中间状态独立。单种子、有限预算的首轮结果只用于定位可行性，不能替代正式模型验收、LLM读图增益或跨任务迁移。

验证：`python -m unittest discover -s experiments/new_tasks_qmap/tests -v`；文字同步：`python -m tasks.export_texts --check`。结论和数值见[报告](results/report.md)。

失败定位：`python -m experiments.new_tasks_qmap.src.diagnose --root runs/new_tasks_qmap/pilot_27893ea --out tmp/new_tasks_diagnosis.json`。该入口只读取保存的数据与预测，不训练或重新选择权重。
