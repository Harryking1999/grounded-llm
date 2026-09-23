# 项目状态与有序 TODO

更新时间：2026-09-23。本页是本分支唯一当前进度页；研究定义见 [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)。

## 当前主线

将 cognitive map／roadmap 作为显式外部 world model：维护实际状态，提供候选动作的预测后果及与目标的地图关系，由 LLM 读取并选择动作。候选距离应从 learned Q-map 的表示计算；真实最短距离只作评测真值或 oracle 上限。图任务先复用已有 Q/V；积木先允许绑定一张初始棋盘，从合法转移学习 Q/V，再检验可解状态与死局是否被区分。

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

1. **完成显式接口的模型对照**：[新实验目录](../experiments/external_map_interface/README.md) 已实现图的距离接口、逐步状态刷新和单棋盘积木 Q/V 训练入口。下一步核对同版本原始／reasoning 模型、服务端、固定任务集与输出预算后，运行匹配的三方对照；真实最短路仅作评测或 oracle。
2. **确认 128 维产物来源**：开发节点当前 SSH 拒绝连接。按已提交 Step 1 配置在本地重放的 official32 地图得到与历史摘要相同的最终距离秩相关 0.9152008119、转移 MSE 4.5962e-11；重放文件在 Git 忽略的 `runs/external_map_interface/replay128/`，与开发机原始文件身份仍须核对。
3. **检验积木 Q/V 的目标几何**：单张 `blocks8_00` 棋盘 pilot 采样 1118 个状态、1134 个不同合法转移；20 轮局部更新后转移 MSE 为 0.0496。193 个明确无合法动作的非空状态中，与构造解上剩余格子更多的可解状态形成 1277 对；其中 780 对的 dead 状态离目标不更远。这是一个负面诊断，不能由转移训练自动推断死局分离。继续分析样本覆盖与目标关系，再决定是否需要调整 Q/V 假设；当前不改变动作或训练目标。

跨图、共享跨任务模型、在线参数学习及其他游戏任务不在本次积木实验范围内。

## 文档与沟通

[Related Work](../RELATED_WORK.md)维护 GCML、状态表征退化与 RAP 的简要摘要、首图及参考意义。任务细节见 [GCML_TASKS.md](GCML_TASKS.md)。
