# 汉诺塔数据

准备入口：任务独立的 `experiments/hanoi_qmap/src/data.py`、`symmetry.py` 与 `run.py`。

状态为每个盘所在柱的向量，盘从小到大；全部关系由环境精确最短路标注，按柱置换关系轨道划分。八盘完整状态图与全关系距离已实现，训练范围见[全图合同](../../../experiments/hanoi_qmap/configs/hanoi8_full_coverage.json)。本目录三盘示例只说明字段，不代表当前盘数；正式评测限定为任意合法起点到指定整塔终点。

`example.json` 是小型结构说明，不是正式训练数据。模型可见输入与监督答案分开；精确标签用于训练和裁判，不作为地图组推理文字。候选编号与真实动作必须逐轮一致。大数据、oracle 表和缓存放在 Git 外，正式划分与参数只放对应实验的 `configs/`。
