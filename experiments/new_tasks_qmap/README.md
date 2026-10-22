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

Token Swapping的短距离关系诊断对应[局部目标训练配置](configs/token_swapping_local_goals_pairs.json)：复用完全相同的精确图与固定留出；保留全部训练源遍历，只混入从当前状态随机游走得到的近目标提案。所有提案仍排除留出端点及关系。它从完整预算基线的best权重开始新优化阶段，优化器与采样器重置，`initialization.json`明确记录身份，不能称作原训练的精确恢复。实际到达率仍按原固定查询验收；增加训练与改变采样的效果不作孤立因果归因。

[Token排序权重对照](configs/token_swapping_local_rank4_pairs.json)复用局部目标阶段相同的基线初始化、种子、数据、采样及预算，仅改变rank权重。保留原验证选择口径，不因训练rank权重改变而重新定义最佳模型；输出、距离函数和推理策略不变。

[rank8对照](configs/token_swapping_local_rank8_pairs.json)由rank4在未见端点固定验证查询上的整题改善触发，只继续改变这一权重；初始化仍从原L∞基线权重开始，不从rank4续训冒充单变量对照。所有正式采用依据使用独特物理查询的冻结重评。

[近目标游走对照](configs/token_swapping_local_walk8_rank4_pairs.json)保持rank4、原初始化、采样混合比例、全源遍历和预算，只缩短局部目标提案的游走上限。提案长度不冒充真实最短距离，实际标签重新查精确表；任意目标提案仍保留。该条件不与rank8同时改动权重。

推箱子的[线性输出诊断配置](configs/sokoban_linear_output_pairs.json)仅替换末端sigmoid，保留两层隐藏MLP、128维Q、有向max、全部数据与监督。此时`cap`仅是输出尺度，向量不再受它约束；不可达仍用配置中的有限代理，不能把线性坐标解释为无穷。其他既有配置默认继续用sigmoid，旧checkpoint不变。

[空间编码器对照](configs/sokoban_spatial_pairs.json)针对输出控制未解决的跨布局失败：从头训练，复用原基线的全部实际数据和采样口径、128维、有向max、Sigmoid、损失、预算及验证选择，只替换状态编码结构。原始四通道棋盘经局部卷积及保留位置的全连接输出Q，不输入目标条件、oracle、死锁规则或路径特征。参数量接近原MLP，精确结构与宽度只维护在配置；这不是激活函数消融，也不混用可达／边界采样实验。旧模型默认结构及state_dict兼容。

[Rush Hour空间编码器对照](configs/rush_hour_spatial_pairs.json)单独检查原MLP的跨布局落差，保留原A–Z身份占据通道及完整目标，改变局部空间编码结构，参数量接近原基线；图、划分、采样、128维、L∞、Sigmoid、损失、预算与验证选择均不变，不将车辆合并成无身份占用图。

[Token分组L1读出对照](configs/token_swapping_grouped_l1_pairs.json)从头复用原L∞基线数据、初始化种子、采样、损失及预算，保留128维独立Q，调用core已有`max_l1`，只改变距离组合。其动机是离线精确L∞不相容状态对见证；不同距离也会改变初始距离与梯度尺度，不能把潜在改善只解释为容量提升。

[Softsign候选配置](configs/sokoban_softsign_output_pairs.json)匹配Sigmoid的中心、中心斜率与范围，先保留为后续对照，未启动。尺度定义、局限及当前结果见[报告](results/report.md#输出与距离尺度)。

[不可达代理尺度对照](configs/sokoban_proxy128_pairs.json)从头保持原Sigmoid基线与原采样，仅改变已证明不可达关系的有限训练代理。新代理仍严格高于接纳的最大有限真距离；实际无穷标签、物理有限标签、输出cap和推理策略不变。它检验代理强迫的大坐标跨度是否影响所学距离场，不把较低MAE当作跨布局导航成功，不与Linear或空间编码混改。

推箱子的[可达与边界采样配置](configs/sokoban_balanced_relations_pairs.json)是独立的数据分布实验，从头训练并保持Sigmoid、同一精确数据／划分、128维、损失、预算及选权重规则。点监督的有限关系与全源遍历配额、边界排序配额及距离分层只由该配置定义；分层内随机抽样。边界指同一状态与目标下同时存在有限和不可达的合法后继。全源配额逐个遍历原训练物理源，因此死路仍进入监督；其余点从精确有限关系银行抽样。排序在边界／非边界关系银行内分层，保留所有合法动作。银行只含原训练源、训练完整目标并排除既有留出关系闭包，能由原数据确定性重建，完整恢复沿用已保存RNG与源遍历状态。该实验不改变正在运行的激活函数对照。

## 冻结评测与信息条件

`src/relation_evaluate.py`报告距离MAE与分层误差、最优候选top-k、方向性、Bellman一致性、不可达识别、贪心到达与最优率、超额成本、循环及具体案例。实际执行轨迹另报预测距离下降／持平／上升、下降余量、非目标零距离及真实最优边比例，检查抽样动作率能否转化为执行中的持续进展。参考决策另核对教师最优边上的预测距离降幅是否接近单位成本1，以及候选选对但距离不下降的数量；排序的正确性不等于预测场的一致性。贪心只比较全部合法后继的`1 + distance(Q(child), Q(goal))`，到精确目标才完成；这些诊断在选动作之后计算，不改变策略。同一批查询比较oracle、简单任务启发式和仅即时成本。

规划评测使用`unique_ordered_state_goal_pairs_v2`：先对完整有序(S,G)去重，再分配布局和距离层预算，不把同源不同完整目标合并。结果保存原始关系行数、去重后的可选题数和实际独特规划题数。旧v1只去重行索引，可能重复同一物理查询，见[核验](results/planning_query_duplicate_audit.json)；原成绩保留来源，修正后各模型冻结重评，不更改训练、回归／排序关系行权重或验证选权重。重复关系行的分层权重不冒充独立规划题数。

距离一致性同时按真实距离区间分别统计参考决策与模型实际访问的执行边，检查访问分布是否比固定参考状态更容易遇到局部极小。曝光数组记录进入损失计算的状态输入次数，不证明每次参数梯度非零；不能把身份对或单候选的零损失也称作有效梯度更新。

推箱子的合法死路仍保留。精确穷举图证明的不可达记为无穷；训练的有限代理单独标识，不声称有限向量等于无穷。资源界导致未完成枚举的场景只记unknown并报告排除原因，不能标作无解。Rush Hour和Token Swapping采用对称距离，跨不同物理场景不定义最短路。

冻结断点可独立复评：

```bash
python -m experiments.new_tasks_qmap.src.relation_evaluate --checkpoint runs/new_tasks_pairs/token_swapping/best.pt --run runs/new_tasks_pairs/token_swapping --out tmp/token_validation.json --cohorts validation unseen_endpoint_validation unseen_both_validation
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/sokoban_pairs.json --run runs/new_tasks_pairs/sokoban --out tmp/sokoban_geometry.json --cohorts validation unseen_goal_validation
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/sokoban_pairs.json --run runs/new_tasks_pairs/sokoban --out tmp/sokoban_actual_batches.json --stream-statistics
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/sokoban_pairs.json --run runs/new_tasks_pairs/sokoban --out tmp/sokoban_gradient_probe.json --checkpoint runs/new_tasks_pairs/sokoban/best.pt --gradient-batches 8
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/token_swapping_pairs.json --run runs/new_tasks_pairs/token_swapping --out tmp/token_linf_witness.json --linf-witness-count 1024
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/token_swapping_local_rank4_pairs.json --run runs/new_tasks_pairs/token_local_rank4 --checkpoint runs/new_tasks_pairs/token_local_rank4/best.pt --out tmp/token_near_endpoint_validation.json --near-endpoint-count 1024
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/token_swapping_local_rank4_pairs.json --run runs/new_tasks_pairs/token_local_rank4 --checkpoint runs/new_tasks_pairs/token_local_rank4/best.pt --out tmp/token_policy_boundaries.json --policy-boundary-count 1024 --cohorts validation unseen_endpoint_validation unseen_both_validation
python -m experiments.new_tasks_qmap.src.relation_diagnose --config experiments/new_tasks_qmap/configs/sokoban_pairs.json --run runs/new_tasks_pairs/sokoban --checkpoint runs/new_tasks_pairs/sokoban/best.pt --out tmp/sokoban_isometries.json --isometry-count 512 --cohorts validation unseen_layout_validation
```

`relation_diagnose`默认用精确距离坐标检查表示能力：当某布局的全部查询目标可放入坐标时，有向max／对称L∞能精确表示到这些目标的距离。这不证明共享MLP已学会地图。目标多于维度时仅检查一组landmark，下界误差不能证明不可表示。这些oracle坐标不用于训练输入、贪心策略或验收成绩。

`--stream-statistics`核对全部实际训练分片，输出逐batch的压缩CSV及汇总：点监督可达／不可达与真实距离层、全合法候选、死锁边界、最优动作、可训练关系覆盖。`--gradient-batches`在固定权重上重放指定checkpoint步之前的实际batch，分别求可达Huber、不可达Huber和排序梯度，不更新模型；保留原总batch分母及rank权重。它不重建未保存的历史逐步梯度。梯度范数不可相加当作贡献百分比，带符号的总梯度方向投影需连同梯度夹角解释。

全训练流统计同时核对“初始源行占位目标的标签”与“在线新目标的真实标签”：基线按前者分层、随后更换目标，二者的距离与可达性不必相同。逐batch转移计数用于定位这一差异，不能把初始分层配额称为真实监督比例。推箱子完整目标探针可使用跨布局统一的100个玩家格坐标；它是oracle依赖的存在性构造，不是学习器或推理特征。

`--linf-witness-count`在固定Token精确图上提出直径状态对，并逐项重算全部交叉距离，验证哪些状态对不能共用一个达到其真距离的L∞坐标。正间隙团给出精确嵌入维数下界及一致绝对误差下界；它不证明平均误差下界或80%贪心不可能，也不用于模型推理。见证保留端点ID和证明，不依赖学习器权重。

`--near-endpoint-count`只生成固定近距验证诊断：真实距离1–5等配额，分别为未见起点、未见目标、两端未见，每对至少一个端点在validation fold，端点不使用test fold。所有模型按同种子重建同一组独特查询，候选仍包含全部合法后继；查询、标签、候选和端点fold保存在旁边的`.queries.npz`。这不更改主验收题集，也不把诊断题写回训练或原固定关系表。

`--policy-boundary-count`只回放指定验证组的冻结贪心，保存每条实际执行边的源／目标、全部合法候选及其预测与真值、所选动作和回环标记，写入`.edges.npz`。汇总分别统计全部访问、首次无严格下降后继、所有非下降访问、闭环动作，并按真实距离分层。分母是实际执行边，同时报告独特状态目标对数量；这些统计不修改动作、剪枝或终止条件，不将验证失败样本加入训练。

`--isometry-count`固定验证查询，检查保持物理距离的八种输入变换：推箱子四通道一起旋转／镜像，Rush Hour同时重命名当前／目标／所有后继的普通车辆且保留A，Token Swapping同时重命名token且固定底图。每个变换对单个状态独立计算，不依赖查询目标。记录距离预测变化、动作一致性及完整贪心；候选顺序沿原查询映射，排除重新枚举造成的平分顺序差异。查询与逐变换预测保存为`.queries.npz`，标签和原验收不变，也不将这些诊断样本加入训练。

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
