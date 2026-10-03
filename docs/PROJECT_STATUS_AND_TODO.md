# 项目状态与有序 TODO

更新时间：2026-10-04。本页是本分支唯一当前进度页；研究定义见 [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)。

## 当前主线

用户已授权实现并启动四 GPU 训练与评测。**优先长轨迹：寻路一张固定图，积木 1000 张初始棋盘；完整轨迹每轮的远近排序、动作及最终终止都学习和评测。**正式参数见[寻路配置](../experiments/flamingo_map_reader/configs/path_single_long.json)、[积木配置](../experiments/flamingo_map_reader/configs/blocks1000_long.json)，解释见[设计](../experiments/flamingo_map_reader/DESIGN.md)。

代码已接入自洽重编号、固定编号重复、完整轨迹缓存和两任务共用的逐轮／闭环评测。数据选择、地图接口、训练、指标和队列分开维护。正式启动前用数据中实际最长训练样本核对显存和反向传播，不能截掉后续轮或终止段。

**助手答案只要求读出远近关系，不再写数值距离。**环境提示与地图记忆始终给出全部合法候选，只有答案收窄到距离最近的 `reported_candidates`（当前为 10）个，排序行与选谁一并收窄；被评的逐对、方向、最近集合等比率都只在被点名的候选上计算。积木候选多（每轮平均 23.8 个，最多 102 个），逐候选数值行和关系句曾把最长样本推到 22,779 token、超出 80 GiB 显存；去掉它们并收窄答案后长度回落到单卡可训练范围。数值距离不再被监督，因此不再报告距离误差，只报告排序、方向、动作与终止。

正式数据准备与启动核验已完成，两条臂已在四卡队列上从头训练，新训练成绩尚未产生。运行记录见[运行位置](#运行位置)，不用旧 16／4 或 swap 结果填补空白。寻路图的真实直径为 11 步；原 1000 棋盘的已认证长任务主要为 7–8 步，复用其物理任务而重新生成所有监督。

**旧结果降为历史：**[原四图完整轨迹及小样本](../experiments/flamingo_map_reader/results/archive/full_trajectory_history.md)、[swap](../experiments/flamingo_map_reader/results/archive/swap_training.md)不参与本轮判断。原工作区自然目标 Q-only 草稿保持原样，不进入本次代码提交或训练。

## Previous approach：连续 state token 对齐（历史实验）

图 Step 2 曾将节点 Q 经 Linear／MLP 映射为冻结 LLM 的连续输入 token，训练节点报告并测一步选择。已执行代码、[runbook](../experiments/cml_map_scaling/STEP2_RUNBOOK.md)、[配置](../experiments/cml_map_scaling/configs/step2.json)及[验收结果](../experiments/cml_map_scaling/results/step2_acceptance.md)保留供复现。固定编号报告达到 100/100，但正确 Q 对错配 Q 的动作选择没有稳定优势；重编号接口在训练中见过的排列上为 0/32。这些结果不支持把 token 对齐继续作为当前主线。

积木 Step 2 曾直接将 10×10 二值棋盘（当时 Q=I）经 MLP 映射为一个 token，训练冻结 Qwen3-4B-Instruct-2507 报告棋盘。原始协议、配置、代码和运行引用保留在[实验目录](../experiments/state_interface_pilot/README.md)；它不再是当前下一步。历史证据包括：固定八张训练棋盘累计拟合到 8/8；多样状态训练后，未见开发棋盘完整报告 0/356，逐格匹配 78.27%，结果与覆盖率捷径相容；单格查询在第 70 轮开发集为 28,338/35,600（79.60%），逐格重建整板仍为 0/356。单格读出不能证明完整状态已可靠供规划使用。此前在读出失败后启动的规划评测也不能用于判断有效 token 的规划价值。详细时间线、诊断和产物路径见[读出报告](../experiments/state_interface_pilot/results/readout.md)。

## 已完成结果

### 历史地图读取接口：完整轨迹与小样本

旧条件、训练设置与选用权重见[历史记录](../experiments/flamingo_map_reader/results/archive/full_trajectory_history.md)。16／4 小样本及原四图结果与本轮新训练分开，不作为当前完成情况。

### Q/V 地图：转移准确，并形成远近关系

四个条件共 64 张地图已完成，覆盖原始 32 节点图及 32–512 节点随机图。所有条件的后继节点识别均为 100%。512 节点图的距离秩相关均值如下：

| 条件 | 距离秩相关 |
| --- | ---: |
| 局部更新，128 维 | 0.559 |
| 局部更新，1000 维 | 0.585 |
| 局部更新，2048 维 | 0.590 |
| 完整梯度，1000 维 | 0.541 |

原始 32 节点图的局部 1000 维结果从初始化的 0.082 升到 0.941。128／256 节点两个 case 的长边诊断表明，二维特别长的连线主要来自投影拉伸；续训到 100 轮没有明显改变高维几何。详见 [Step 1 报告](../experiments/cml_map_scaling/results/report.md)。

开发节点已重新连通，原始 `local128/official32` 产物可读取：Q 为 32×128、V 为 96×128，最终距离秩相关 0.9152008119、转移 MSE 4.5962e-11，与此前重放及历史摘要一致。

### 积木 Q/V：转移拟合与目标距离分离

单棋盘 `blocks8_00` 的共享动作向量试验已按同样 100 轮分别采 200 和 2,000 次合法 rollout。扩采地图含 8,478 个状态、9,794 条不同合法转移，转移 MSE 为 0.000226；但在 9,676 对“较少格子的立即死局 vs 构造解上的可解状态”中，9,509 对死局的 learned Q 距离不更远。在两张地图共同覆盖的固定 1,277 对中，扩采地图仍有 1,255 对不更远。它表明本次训练虽拟合转移，却尚未学出所需的规划距离；两次地图初始化不完全相同，不能将排序变化单独归因于采样量。状态覆盖和完整数值见[积木 Q/V 试验结果](../experiments/external_map_interface/results/blocks_q_pilot.md)。

### 显式地图距离：path256 逐步评测

同一固定图上的 Qwen3-4B 四组各 128 条已完成并通过原裁判重放验收。非 thinking、无地图 `plain` 最短路 0、到达 69；非 thinking、地图距离 `plain_distance` 最短路 8、到达 90；thinking、无地图 `reasoning` 最短路 7、到达 10；thinking、地图距离 `distance` 最短路 0、到达 4。后两组分别有 117、124 条因某一步耗尽 16,384 输出 token 截断。非 thinking 配对在这张固定图上观察到地图收益，但 8 条最短路仅来自 2/16 道题；不能宣称跨图稳定。固定图的 128 维地图距离秩相关 0.575694；只按地图距离贪心的只读诊断在 16 题中到达 16、最短 4。详见 [path256 显式距离报告](../experiments/external_map_interface/results/path256_distance.md)。

另一张独立 256 节点图按最短路长度 4、6、8、10 各取 4 题，重新训练 128 维 Q/V 后，Qwen3-4B 非 thinking 的无地图／地图组到达为 53/128→97/128，最短路均为 4/128。两图合计到达 122/256→187/256，最短路 4/256→12/256；到达提升方向跨两图一致，最短路总数改善只见于原图。新图地图距离秩相关 0.515，地图贪心到达 16/16、最短 5/16。Qwen3-4B-Instruct-2507 同题两组均 0/128，主要因回答不符合单动作 JSON 格式或单步截断，尚不能评估其地图规划收益。四组均已重放裁判；详见[独立图复测报告](../experiments/external_map_interface/results/path256_diverse.md)。

五图正式评测已完成并通过全量重放：每组 120 题、每题 16 次，共 5,760 条轨迹。三组使用[冻结题集](../experiments/external_map_interface/configs/path256_five_graphs_suite.json)和[评测合同](../experiments/external_map_interface/configs/path256_five_graphs_eval.json)，统一从正常结束回答的末尾提取动作。23,508 次正常结束回答均成功提取；样本槽位、合同、seed、prompt、状态、动作转移和原裁判结果没有不一致。

| 条件 | 最短路 pass@1 | pass@8 | pass@16 | 到达／1,920 |
| --- | ---: | ---: | ---: | ---: |
| Instruct-2507 | 26.98% | 67.13% | 79.17% | 691 |
| Instruct-2507＋地图 | 33.49% | 70.34% | 80.00% | 846 |
| Thinking-2507 | 47.97% | 79.50% | 86.67% | 1,140 |

同权重地图对照的 pass@1 提升 6.51 个百分点，五图方向一致；pass@8 提升 3.20 点，但按题配对重采样区间仍跨零；pass@16 仅多覆盖 1 题（95→96/120）。10 步题是负结果：地图组最短路从 28/480 降为 24/480。地图组每条轨迹平均输出减少 8.2%，两组都成功的 317 对轨迹减少 4.2%；计入失败尝试后，每次最短路成功的累计输出成本减少 26.0%。轨迹截断率从 63.28% 降为 55.00%。这些是成绩与生成成本的改善；当前没有 token 概率，尚不能归因于低熵变化。Thinking 使用另一份权重，单独解释其成绩。详细分析见[五图报告](../experiments/external_map_interface/results/path256_five_graphs.md)，完整紧凑数值见[结果摘要](../experiments/external_map_interface/results/path256_five_graphs.json)。旧两图结果作为历史背景，不混入此次统计。

### Qwen thinking 基线：寻路随规模改善，积木仍未成功

| 条件 | 4B | 8B | 32B |
| --- | --- | --- | --- |
| 双向 path256 最短解 | 8/128 | 15/128 | 35/128 |
| 同批输出合法到达 | 43/128 | 57/128 | 64/128 |
| 最短解 pass@8 | 7/16 | 10/16 | 11/16 |
| Blocks 8 块 | 0/128 | 0/128 | 0/128 |
| Blocks 12 块 | 0/128 | 0/128 | 0/128 |

Blocks 共 768 次，511 次触及预算；至少 343 条已确认非法，6 条合法未清空，419 条截断原因未决。状态报告正确却下一步违规、重复移除和终止判断错误等现象均有记录。原寻路提示要求最短，不能将其到达率当作新“只需到达”条件。详见[验收报告](../experiments/qwen_path_blocks/results/report.md)。

### 其他已完成基线

| 研究 | 关键结果 | 结果入口 |
| --- | --- | --- |
| Sol 新形状积木与 DAG | 8 块 91/128，12 块 69/128，DAG 119/128；pass@8 为 16/16、13/16、16/16 | [报告](../experiments/sol_dag_blocks/results/report.md) |
| Luna 反例批次 | 512 次；积木 99/128、72/128；原图／单开关／双开关最短解 116/128、60/64、40/64 | [摘要](../experiments/gcml_counterexamples/results/summary.json)、[案例](../experiments/gcml_counterexamples/results/luna_case_studies.md) |
| DeepSeek Flash 同题对照 | 512 次含 68 次截断；64 个实例均至少成功一次，完整积木答案全部合法清空 | [摘要](../experiments/gcml_counterexamples/results/deepseek_flash.json) |
| Sol path256 小批试点 | 6 次请求得到 3 条有效答案，均为最短路；另 3 次网关错误 | [摘要](../experiments/qwen_path_blocks/results/sol_path256_pilot.json) |

这些结果来自不同提示、预算和模型设置，按各自实验解释。Sol 小批不能估计稳定准确率；Luna 难例也不是多个模型共同的稳定失败。各实验目录保留正式配置与数值来源。

## 有序 TODO

1. 完成正式数据生成，核对长轨迹、全部棋盘、固定编号版本及测试任务实际覆盖；配对不足与贪心失败如实记录。
2. 在真实最长样本通过显存与梯度检查后，从头训练两份完整轨迹接口；四卡队列同步消费阶段评测。
3. 完成最终逐轮、闭环、编号重排、真实目标替换及无地图评测。分别报告排序、方向、动作、终止和成功率，按轮次、候选数量及剩余距离分层。
4. 根据新结果解释训练内是否学会、同图／同板新任务是否泛化及地图是否发挥作用。未完成的结果不先下结论。

### 运行位置

远端根目录为 `/zhanghanyue/experiment/flamingo_map_reader`。当前运行使用独立提交的源码目录及新的运行目录：

- 源码 `code/long_f73b700`（分支 `codex/long-trajectory-training`，提交 `f73b700`）；运行目录 `runs/long_f73b700_20261004`
- 队列 pid 4925（`--gpus 0 1 2 3 --prepare-workers 24`），日志 `runs/long_f73b700_queue.log`；寻路训练 pid 5005（GPU 0），积木训练 pid 5100（GPU 1）
- 阶段评测由队列按稀疏日程消费 GPU 2–3：`checkpoint-0` 2 条、`checkpoint-128` 256 条训练诊断、每个整轮边界加全量 validation，`final` 才跑完整评测（test ×6 编号变体与 no_map 对照）。其余 checkpoint 不产生评测任务，因此 GPU 2–3 多数时间空闲，这是有意让评测不扰动训练。

Smoke 用各自真实最长的训练样本跑完整轨迹、不截断：

| 任务 | 最长 token | 峰值显存 | 梯度检查 |
| --- | ---: | ---: | --- |
| 寻路 | 11,451 | 51.85 GiB | 有限；地图编码器无梯度 |
| 积木 | 8,960 | 41.70 GiB | 有限；地图编码器无梯度 |

实际覆盖（train／validation／test）：寻路 4192／532／1032 个物理任务，编号后 24192／3032／6032 条，最长 11,451／10,723／11,186 token，贪心全部成功；积木 9000／1100／2200 个物理任务，编号后 49000／6100／12200 条，最长 8,964／9,138／8,228 token，贪心成功 9000／783／1549，未成功的任务保留在数据中并如数记录。

未训练地板（`checkpoint-0`，2 个任务、10 个决策轮）：6 轮给出可解析排序行，`exact_ranking` 0，合法动作 7/10，从不主动终止。这是地板而非成绩。

旧运行 `long_9234c5c_20261003`（旧协议，已停止）及其权重保留供对照，不参与本轮判断。

## 文档与沟通

[Related Work](../RELATED_WORK.md)维护 GCML、状态表征退化与 RAP 的简要摘要、首图及参考意义。任务细节见 [GCML_TASKS.md](GCML_TASKS.md)。
