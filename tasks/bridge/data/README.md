# 手电过桥数据

每个场景由人数与完整耗时组合定义，按排序后的耗时元组划分，训练、开发与独立验收组合排除重叠。五人、六人分别准备与训练；耗时为配置范围内互异正整数，升序定义人员编号。参数只维护在[selection所指配置](../Qmap/selection.json)。

每场景枚举全部人员位掩码与手电岸，保留合法载1／2人的全部动作及即时耗时；对选定完整目标运行Dijkstra，生成距离与完整候选。目标包括全员右岸和指定人员／手电中间分布，起点可为全员左岸或中间状态。无解真值保留为无穷，训练代理值另存。

生成实现见[relation_data.py](../../../grounded_llm/tasks/bridge/relation_data.py)。精确图、曝光、日志和断点保存在忽略运行目录；[报告](../../../experiments/new_tasks_qmap/results/bridge/report.md)链接已采用地图的独立题集结果。[example.json](example.json)仅作格式说明；早期pilot题集单列存档。
