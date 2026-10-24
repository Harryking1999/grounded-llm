# Qmap 历史公共实现与兼容入口

当前工作区按任务拆分：汉诺塔使用 `experiments/hanoi_qmap/src/run.py`，魔方使用 `experiments/cube2_qmap/src/run.py` 及其专属续训脚本，寻路使用 `experiments/path_qmap/src/run.py`。本目录保留历史训练与导入兼容，不再作为当前混合训练入口。任务入口见 [tasks](../../tasks/README.md)，统一方法与基础组件见 [QMAP_METHOD](../../tasks/QMAP_METHOD.md)。以下内容保留原研究设计语境。

目标是让网络通过距离和动作关系，自行形成有远近关系的状态空间。每个状态只对应一个与目标无关的向量 Q(s)；面对不同目标时，使用同一组坐标和固定距离规则。

```text
状态 → 任务输入编码 → 共享结构的 MLP → Q(s)
合法后继与目标分别编码 → 比较向量距离 → 选择最近的后继
```

Q 在这里是状态向量，不是强化学习的动作价值 Q(s,a)。网络不接收人工指定的坐标答案。精确最短路只用于训练标签与评测，求解时不查距离表。

积木、寻路、汉诺塔和魔方共用 `StateEncoder` 的两层隐藏 MLP 与 Q 输出结构，并共用软距离排序函数；只适配输入宽度，固定距离由任务配置指定。模型分别训练，四个任务均可向 cross-attention 提供同格式的当前、目标和真实合法后继 Q 槽位；统一接口见 `flamingo_map_reader/src/state_q.py`。

训练同时约束两件事：向量距离接近真实最短步数；全部合法候选到目标的距离排序正确。前者提供远近尺度，后者连接动作选择。当前状态和后继由环境提供。

## 阅读入口

| 内容 | 入口 |
|---|---|
| 魔方的回环、对称性与距离选择 | [魔方说明](../cube2_qmap/README.md) |
| 距离的数学限制与相关工作 | [结构说明](../cube2_qmap/results/structure_notes.md) |
| 汉诺塔任务 | [任务说明](../hanoi_qmap/README.md) |
| 寻路的节点独热输入与统一 Q 训练 | [任务说明](../path_qmap/README.md) |
| 当前工作与下一步 | [项目状态](../../docs/PROJECT_STATUS_AND_TODO.md) |
| 当前模型与冻结读取接口 | [汉诺塔](../hanoi_qmap/src/model.py)、[魔方](../cube2_qmap/src/model.py) |
| 当前评测 | [汉诺塔](../hanoi_qmap/src/evaluate.py)、[魔方](../cube2_qmap/src/evaluate.py)；[历史命令行求解](src/solve.py) |

参数以各任务 configs/ 为准。不同 checkpoint 可以使用不同固定距离，日志、数据和权重放在 Git 外；未取得目标成功前，文档不追加逐次实验记录。
