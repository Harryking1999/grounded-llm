# 二阶魔方任务工作区

后继地图实验采用[selection](Qmap/selection.json)的 `goal_relative_inverse` 条件：目标相对坐标与编码器逆对称，全状态完成94.90%，Flan同题复原117/125、非复原目标119/125。已完成的Flan 4／5／6类实验仍使用旧 `repair_guard` 权重和绝对状态编码；新旧结果见[实验报告](../../experiments/cube2_qmap/results/report.md)。新条件已验收地图贪心，SFT入口的目标坐标槽位接入仍待实现，不能直接替换旧缓存。

- [文字、规则与各组提示](prompts/README.md)：任务内原文；根目录 `TASK_TEXTS.md` 是自动集中副本。
- [数据准备与示例](data/README.md)：小例说明字段。
- [Qmap](Qmap/README.md)：训练、加载、选定权重与适用范围。
- [环境规则](../../experiments/cube2_qmap/src/environment.py)；[研究协议与实现](../../experiments/cube2_qmap/README.md)。
- 当前进展与下一步只维护在 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

按本任务独立准备数据、训练和评测。共用 MLP、距离与排序损失等基础组件；不同任务的状态编码、动作、采样与训练入口分别维护。
