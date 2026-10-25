# 手电过桥 Qmap

状态编码含各人的时间、所在岸和手电岸；使用共享 MLP 与对称 L∞ 距离。时间是场景上下文，不是目标信息。目标指定同场景完整人员分布和手电岸。动作评分为本次耗时＋地图后继距离。

训练复用 [StateEncoder、distance、soft_rank_loss](../../../grounded_llm/core/qmap/model.py)，使用距离Huber与完整候选排序；五人沿用软成本目标，六人所选方案将排序目标均分到所有真实最优动作。具体输入编码、输出形式与训练条件由采用配置给出。不可达以配置中有限代理值训练与排序；它不是精确无穷距离。评测始终保留合法死路，并报告成本、动作顺序和完整求解。

五人、六人分别训练的地图均通过各自冻结后的独立完成与成本验收；六人的早期失败条件仍保留。采用权重、人数和耗时输入范围只维护在[selection](selection.json)，证据见[过桥报告](../../../experiments/new_tasks_qmap/results/bridge/report.md)。`grounded_llm.tasks.bridge.qmap.MapReader`只读取checkpoint和物理状态特征，返回实际过桥人员动作，不依赖精确图或距离表。LLM读取尚未验收。
