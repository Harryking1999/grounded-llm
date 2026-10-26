# 21／24点

采用范围为**1–10任取四个不同整数的全部210组起题**，目标为21、24或配置内完整中间数值多重集。默认多目标地图四数→21完成185/185、→24完成196/197；留出两步4→2目标348/420。旧单数目标专用地图保留，两个版本范围见[采用记录](Qmap/selection.json)。优化已暂停，尚无本任务LLM读取验收。

状态是剩余数的精确有理数多重集；一步选择两个数与加减乘除之一并合并，所有数必须用完，不能丢弃。目标是剩余数恰好等于指定多重集。每步元素数减1，所以可达时步数固定，主要考查保留解的分支，而不是长程循环。

四数及其2／3数后缀共用四槽Qmap。210组全部在训练域，不能宣称未见起题泛化；独立3数初题和5数地图未采用。规则与裁判已实现，5槽动作支持也不代表有5数训练权重。

- [规则与输出](prompts/README.md)
- [数据与覆盖](data/README.md)、[格式示例](data/example.json)
- [输入与读取](Qmap/README.md)、[采用记录](Qmap/selection.json)
- [当前成绩、边界及关键比较](../../experiments/new_tasks_qmap/results/game24/report.md)
- [环境与裁判](../../grounded_llm/tasks/game24/environment.py)

## 来源与适配

参考[XoT Game24环境](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_mcts/game24/Game24Game.py)。四数动作顺序保留，以精确有理数和显式长度替代浮点数与近似判等；不使用其MCTS或学习权重。早期XoT起题划分及固定24试验是历史条件，当前有限域另行穷举生成。
