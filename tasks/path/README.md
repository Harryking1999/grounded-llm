# 寻路任务工作区

采用魔方／汉诺塔分支中新增的统一方法寻路 Q（`experiments/path_qmap`），不采用旧 Q/V。当前合同是固定图、节点独热输入；已采用本方法 path256_01c053f 的验证最优 best.pt；测试到达 251/256（98.05%）。

- [文字、规则与各组提示](prompts/README.md)：任务内原文；根目录 `TASK_TEXTS.md` 是自动集中副本。
- [数据准备与示例](data/README.md)：小例说明字段，不冒充正式训练集。
- [Qmap](Qmap/README.md)：训练、加载、选定权重与适用范围。
- [环境规则](../../experiments/external_map_interface/src/transitions.py)；[研究协议与实现](../../experiments/path_qmap/README.md)。
- 当前进展与下一步只维护在 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

按本任务独立准备数据、训练和评测。共用 MLP、距离与排序损失等基础组件；不同任务的状态编码、动作、采样与训练入口分别维护。
