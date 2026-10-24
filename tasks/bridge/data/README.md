# 手电过桥数据

生成入口：[data.py](../../../grounded_llm/tasks/bridge/data.py)。参数和划分数量只维护在[配置](../../../experiments/new_tasks_qmap/configs/pilot/bridge.json)。运行输出为忽略目录中的 data.npz、scenes.jsonl.gz 和 data_summary.json，包含场景、状态编码、全部合法动作后继、动作成本、精确距离、终止掩码、起点及划分。不可达距离是正无穷，训练代理标签另存，不覆盖真值。

按排序后的完整时间组合划分，训练、验证、测试组合不重叠；每场景完整枚举人员位置和手电位置，初始均全员左岸。经典四人题只作规则回归。

[example.json](example.json)仅为结构与规则说明。
