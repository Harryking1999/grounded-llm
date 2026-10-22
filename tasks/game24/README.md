# 24点

独立规则、数据适配和裁判已实现；首轮仅检验固定目标 Qmap 可行性。正式范围与预算见[配置](../../experiments/new_tasks_qmap/configs/game24.json)，结果见[报告](../../experiments/new_tasks_qmap/results/report.md)。不作为多任务SFT已采用任务。

- [规则与输出](prompts/README.md)
- [数据](data/README.md)与[示例](data/example.json)
- [地图](Qmap/README.md)与[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/game24/environment.py)

## 来源与适配

参考 [XoT Game24 环境](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_mcts/game24/Game24Game.py) 和该提交的 train/test CSV。保留36个两数运算动作的顺序，用精确有理数和显式长度替代浮点数、0.0001填充及1e-4判等；不使用其MCTS或学习权重。
