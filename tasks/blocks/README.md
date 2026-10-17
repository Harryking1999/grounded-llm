# 积木任务工作区

现有积木同数据多基线已完成，结果及 B2_10k 异常诊断见原实验报告。任务文字沿用本轮真实实现；新的难度与正式强基线预算待确认。

- [文字、规则与各组提示](prompts/README.md)：任务内原文；根目录 `TASK_TEXTS.md` 是自动集中副本。
- [数据准备与示例](data/README.md)：小例说明字段，不冒充正式训练集。
- [Qmap](Qmap/README.md)：训练、加载、选定权重与适用范围。
- [环境规则](../../experiments/gcml_counterexamples/src/blocks.py)；[研究协议与实现](../../experiments/blocks_distance_map/README.md)。
- 当前进展与下一步只维护在 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

按本任务独立准备数据、训练和评测。共用 MLP、距离与排序损失等基础组件；不同任务的状态编码、动作、采样与训练入口分别维护。
