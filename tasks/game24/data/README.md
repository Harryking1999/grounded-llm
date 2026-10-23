# 24点数据

生成入口：[data.py](../../../grounded_llm/tasks/game24/data.py)。参数和划分数量只维护在[配置](../../../experiments/new_tasks_qmap/configs/pilot/game24.json)。运行输出为忽略目录中的 data.npz、scenes.jsonl.gz 和 data_summary.json，包含场景、状态编码、全部合法动作后继、动作成本、精确距离、终止掩码、起点及划分。不可达距离是正无穷，训练代理标签另存，不覆盖真值。

来自固定提交XoT CSV。先按无序四数去重，移除训练源中与测试源重合的题，再从官方训练源划分训练和验证；官方测试题顺序保留。完整展开三步有向状态图。不同起题可能共享中间状态，因此另外报告测试状态编码在训练中的重复数及未见状态动作正确率，不能把题目留出等同于状态留出。

[example.json](example.json)仅为结构与规则说明，不作为独立评测题。
