# 项目状态与有序 TODO

更新时间：2026-09-27。本页是本分支唯一当前进度页；研究定义见 [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)。

## 当前主线

将 cognitive map／roadmap 作为显式外部 world model：维护实际状态，提供候选动作的预测后果及与目标的地图关系，由 LLM 读取并选择动作。推理阶段的候选距离从 learned Q-map 计算。图任务复用已有 Q/V；积木已获用户授权改为直接监督最短距离／不可达关系、先只训练 Q，并改用原论文八种简单形状。精确最短距离在该新方案中可作离线训练标签，但不能混作 learned-map 推理输出。

当前只研究寻路与积木。首版已选仅暴露 learned-map 候选距离；寻路优先复用已有 128 维 Q/V。完整 Q 与较短 Q 向量仍是可能的后续接口版本。每步以环境执行后的实际状态刷新上下文。高复杂度下的多路径采样留待基本地图、接口和规划跑通之后。唯一有序工作见下文 TODO。

## Previous approach：连续 state token 对齐（历史实验）

图 Step 2 曾将节点 Q 经 Linear／MLP 映射为冻结 LLM 的连续输入 token，训练节点报告并测一步选择。已执行代码、[runbook](../experiments/cml_map_scaling/STEP2_RUNBOOK.md)、[配置](../experiments/cml_map_scaling/configs/step2.json)及[验收结果](../experiments/cml_map_scaling/results/step2_acceptance.md)保留供复现。固定编号报告达到 100/100，但正确 Q 对错配 Q 的动作选择没有稳定优势；重编号接口在训练中见过的排列上为 0/32。这些结果不支持把 token 对齐继续作为当前主线。

积木 Step 2 曾直接将 10×10 二值棋盘（当时 Q=I）经 MLP 映射为一个 token，训练冻结 Qwen3-4B-Instruct-2507 报告棋盘。原始协议、配置、代码和运行引用保留在[实验目录](../experiments/state_interface_pilot/README.md)；它不再是当前下一步。历史证据包括：固定八张训练棋盘累计拟合到 8/8；多样状态训练后，未见开发棋盘完整报告 0/356，逐格匹配 78.27%，结果与覆盖率捷径相容；单格查询在第 70 轮开发集为 28,338/35,600（79.60%），逐格重建整板仍为 0/356。单格读出不能证明完整状态已可靠供规划使用。此前在读出失败后启动的规划评测也不能用于判断有效 token 的规划价值。详细时间线、诊断和产物路径见[读出报告](../experiments/state_interface_pilot/results/readout.md)。

## 已完成结果

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

1. **积木 Q-map 距离监督试点（当前执行）**：用户已授权在开发机训练，并指定原论文八种简单形状。已形成[面向人的方案](BLOCKS_QMAP_PLAN.md)、[agent 执行说明](../experiments/blocks_distance_map/README.md)和[正式合同](../experiments/blocks_distance_map/configs/pilot.json)。精确标签器、分组留出、仅训练 Q 的距离／排序目标及评测已实现；七项正确性检查通过，两个距离形式的短 GPU smoke 已跑通。下一步提交源码与合同，运行同数据、同预算的有向／欧氏 Q 对照并报告有限距离、深层死局、留出目标动作排序。本轮不训练 V，不接入 LLM。
2. **讨论寻路下一轮**：五图评测、全量重放、生成成本与首次失误归因已完成。仅按地图距离贪心的诊断得到 69/120 最短解、120/120 到达，显示当前 LLM 单次表现尚未稳定兑现地图信息。下一轮建议优先讨论小批、两组匹配的受约束 JSON 单动作解码，区别于本次仅在 prompt 中要求格式的条件；同时针对 10 步题定位候选排序错误，之后再讨论 Thinking＋地图。新评测条件与预算待用户确认。

跨图、共享跨任务模型、在线参数学习及其他游戏任务不在本次积木实验范围内。

## 文档与沟通

[Related Work](../RELATED_WORK.md)维护 GCML、状态表征退化与 RAP 的简要摘要、首图及参考意义。任务细节见 [GCML_TASKS.md](GCML_TASKS.md)。
