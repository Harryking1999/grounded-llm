# 项目状态与有序 TODO

更新时间：2026-09-16。本页是唯一当前进度页；研究定义见 [RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)。

## 当前主线

**Step 1 地图学习完成；Step 2 runbook 与配置已确定，接口代码和训练尚未开始。**

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

1. **完成 Step 2 接口实现**：生成报告数据，接入连续状态 token，落实答案序列 loss 与裁判，做小批反传检查并测成本。
2. **训练与评测 Linear／MLP**：只训练报告，分别测读取、一步动作与适配前后几何；完整流程见 runbook。当前未启动实验。
3. **根据失败定位追加条件**：先判断问题在读取还是使用，再决定多 token、编号重排或少量动作监督。
4. **推进多步解题**：单步接口确认后，再定义外部状态更新、动作边界、终止和预算，评估合法到达与效率。

[积木状态接口](../experiments/state_interface_pilot/README.md)保留为后续设计；跨图、共享跨任务模型、在线参数学习及其他游戏任务暂缓。

## 文档与沟通

[Related Work](../RELATED_WORK.md)维护 GCML 和状态表征退化论文的摘要、首图及参考意义。任务细节见 [GCML_TASKS.md](GCML_TASKS.md)。每周五前汇总，日常更新用文字，重大研究决策再讨论。
