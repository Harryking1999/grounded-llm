# 24点

独立规则、数据适配和裁判已实现，首轮地图完成70/137题，尚未采用权重或准备正式SFT数据。它是扩充运算类任务的候选，现有[配置](../../experiments/new_tasks_qmap/configs/pilot/game24.json)、[结果](../../experiments/new_tasks_qmap/results/report.md)与[文字筛查](../../experiments/task_text_baselines/results/report.md)保留原条件，下一轮安排见[项目状态](../../docs/PROJECT_STATUS_AND_TODO.md)。

当前状态是剩余有理数的多重集合，动作是选择两个槽位和一个运算符并合并，目标为只剩24。运算符本身不构成完整动作，操作数选择同样重要。现有实现不依赖表达式历史；扩展到其他运算任务时，应先明确哪些信息决定后续合法动作与目标，复用这些规则和裁判入口。

- [规则与输出](prompts/README.md)
- [数据](data/README.md)与[示例](data/example.json)
- [地图](Qmap/README.md)与[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/game24/environment.py)

## 来源与适配

参考 [XoT Game24 环境](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_mcts/game24/Game24Game.py) 和该提交的 train/test CSV。保留36个两数运算动作的顺序，用精确有理数和显式长度替代浮点数、0.0001填充及1e-4判等；不使用其MCTS或学习权重。
