# 手电过桥 Qmap

状态编码含各人的时间、所在岸和手电岸；使用共享 MLP 与对称 L∞ 距离。时间是场景上下文，不是目标信息。目标指定同场景完整人员分布和手电岸。动作评分为本次耗时＋地图后继距离。

训练复用 [StateEncoder、distance、soft_rank_loss](../../../grounded_llm/core/qmap/model.py)，保持距离Huber＋原权重软排序；按任务独立运行[首轮实验](../../../experiments/new_tasks_qmap/README.md)。不加入任务专属学习损失。不可达以配置中有限代理值训练与排序；它不是精确无穷距离。评测始终保留合法死路，并报告成本、动作顺序和完整求解。

五人地图通过冻结后的独立验收；六人地图因成本最优率不足仍未采用。采用权重、人数和耗时输入范围只维护在[selection](selection.json)，证据见[过桥报告](../../../experiments/new_tasks_qmap/results/bridge/report.md)。`grounded_llm.tasks.bridge.qmap.MapReader`只读取checkpoint和物理状态特征，返回实际过桥人员动作，不依赖精确图或距离表。LLM读取尚未验收。
