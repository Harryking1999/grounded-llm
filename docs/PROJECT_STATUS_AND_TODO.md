# 项目状态与有序 TODO

更新时间：2026-09-21。本页是本分支唯一当前进度页；研究定义见 [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)。

## 当前主线

**当前分支推进 Blocks Step 2：文字状态与文字加一个连续状态 token 的严格逐步配对实验。**

用户已确认 Qwen3-4B-Instruct-2507；不依赖地图 Step 1，二值棋盘直接输入 adapter。
每步必须给出动作和动作后完整自报棋盘；任一错误立即失败，只有双重正确才追加状态 token。
正式配置见 [blocks_step2.json](../experiments/state_interface_pilot/configs/blocks_step2.json)，
执行说明见 [积木接口](../experiments/state_interface_pilot/README.md)。使用原积木规则，覆盖八块及较低构造块数，不测十二块。

已复用另一开发分支 `cdd9e58` 的功能化 harness，并补充像素接口和严格逐步裁判；四项协议测试通过。
真实模型 smoke 待执行；训练和正式配对评测尚未启动。
两台开发机已确认可达；用户最新告知七张 GPU 均可用。运行位置与成本在实际启动后记录。

下列图任务计划是本分支 2026-09-16 的历史快照；其他工作树已推进后续实验，不能将其视为全项目最新运行状态。

- 使用 Qwen3-4B-Instruct-2507，冻结 Q/V 和 LLM，只训练 Linear／MLP 报告当前与目标节点。
- 统一提供文字图结构，加入准确文字状态对照；动作选择只作训练后评测。
- 多 token、打乱编号、少量动作监督均留作后续解决方案。
- 当前执行入口：[runbook](../experiments/cml_map_scaling/STEP2_RUNBOOK.md)、[配置](../experiments/cml_map_scaling/configs/step2.json)。原讨论稿保留参考，参数以配置为准。

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

1. **Blocks GPU smoke**：真实反传检查 loss、冻结参数与像素槽梯度；验证完整 JSON 停止和同序列续写。
2. **训练状态报告 adapter**：复用共享训练器，按开发集读出选模，不加入动作监督。
3. **配对逐步解题**：同题文字／文字加 token，记录成功数、合法动作数、正确报告前缀、失败类型及实际预算。
4. **解释结果**：结合独立读出准确率区分接口未学会与规划未受益，再决定下一轮；不自动扩大条件。

跨图、共享跨任务模型、在线参数学习及其他游戏任务不在本次积木实验范围内。

## 文档与沟通

[Related Work](../RELATED_WORK.md)维护 GCML 和状态表征退化论文的摘要、首图及参考意义。任务细节见 [GCML_TASKS.md](GCML_TASKS.md)。每周五前汇总，日常更新用文字，重大研究决策再讨论。
