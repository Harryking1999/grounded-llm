# 寻路 Qmap

训练与加载：`experiments/path_qmap/src/run.py`；加载 `FrozenPathMap`。只使用本次统一方法的寻路 Q。

共享基础：两层隐藏 MLP、128 维 Q、距离监督与候选排序，见 `tasks/QMAP_METHOD.md`；任务状态编码、数据、训练与评测分别维护。参数只在正式合同中定义，见 `selection.json` 的 `config`。

选定模型来源与适用条件记录在 [selection.json](selection.json)。选定权重放 `checkpoints/`，由 Git 忽略；过程断点与原始日志仍留在运行／归档目录。不用失败续训覆盖已采用模型。

采用魔方／汉诺塔分支中新增的统一方法寻路 Q（`experiments/path_qmap`），不采用旧 Q/V。当前合同是固定图、节点独热输入；已采用本方法 path256_01c053f 的验证最优 best.pt；测试到达 251/256（98.05%）。

FLAN类内OOD另提供一张同规模、同度分布和同密度的新图及其独立Qmap，采用身份见 selection 的 `additional_maps.flan_new_graph256`。新图关系测试到达248/256（96.88%），见[结果](../../../experiments/path_qmap/results/report.md#flan新图)。LLM读取待评测。
