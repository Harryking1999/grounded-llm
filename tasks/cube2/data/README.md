# 二阶魔方数据

准备入口：任务独立的 `experiments/cube2_qmap/src/relation_data.py`、`data.py`；复原数据在 `fixed_goal.py`，当前修复续训在 `cycle_repair.py`。

角块置换与朝向编码；精确距离来自对应动作制的离线 oracle，完整候选后继与到目标距离作为监督。当前 JSON 的 null 明确表示尚未填写的文字／数据示例，不可用于训练。

`example.json` 是小型结构说明，模型可见输入与监督答案分开；精确标签用于训练和裁判，不作为地图组推理文字。候选编号与真实动作必须逐轮一致。大数据、oracle 表和缓存放在 Git 外，正式划分与参数只放对应实验的 `configs/`。
