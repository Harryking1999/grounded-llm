# 二阶魔方 Qmap

训练与加载：`experiments/cube2_qmap/src/run.py`、`relation_data.py`、`model.py`、`evaluate.py`；当前采用权重的续训入口为同目录 `cycle_repair.py`，复原为 `fixed_goal.py`。不调用汉诺塔训练器。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。采用权重与过程断点分别保存。

当前采用循环修复对照的 `repair_guard` 复原权重，保留对称 Q 与 L∞。绝对状态编码的非复原目标能力仍弱；目标相对坐标的冻结Q读取结果见[实验报告](../../../experiments/cube2_qmap/results/report.md)。

读取入口为 `grounded_llm.tasks.cube2.model.MapReader`。默认 `coordinate_frame="absolute"`；`coordinate_frame="goal_relative"` 在 `scores`／`choose` 中使用已知群运算转换到复原问题。两者共享同一权重，目标相对坐标条件单独标注。

新结构在模型配置中启用 `inverse_symmetry=true`，编码器内部对正反相对状态的Q向量求均值，严格保证 `Q(r)=Q(r⁻¹)`。仍输出128维，读取使用原L∞距离，并以 `coordinate_frame="goal_relative"` 加载。编码器内部处理两种对称呈现；历史距离均值读取 `inverse_average=True` 的对照见实验报告。

后续目标相对读取采用selection的`goal_relative_inverse`条件：验证选出的第486轮权重，全状态94.90%，四组固定题分别93.2%、93.4%、95.2%、94.4%；当前FLAN仍使用原冻结权重和绝对读取。
