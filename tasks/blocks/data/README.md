# 积木数据

Qmap正式清空验收使用[固定1000题](clear1000.json)：原八形状、每盘12块构造、目标为空。所有模型使用相同题目和合法动作；动作上限为初始占格数除以2取整，规则与[验收合同](../../../experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json)一致。该题集不进入训练，原文字40题与历史SFT510题分别保留，不能互相替代。

`example.json` 是用真实规则构造的一步可解说明例，有非空目标，不是新增训练样本。`prepare_example.py` 调用当前 SFT 的真实提示函数生成例子和两组逐轮文本：

```bash
python -m tasks.blocks.data.prepare_example
python -m tasks.export_texts
```

正式数据：原始 HDF5 棋盘→来源与留出划分→物理轨迹→地图／无地图监督文本→按各模型 tokenizer 编码的缓存。真实准备实现是 `experiments/map_sft_baselines/src/prepare.py`；10k 分层抽取是同目录 `subsample.py`，7B 同轨迹重编码是 `retokenize_plain.py`。

轨迹字段包括 `trajectory_id`、`board_row`、十进制字符串 `start/goal`、`shortest_moves`、每轮局部编号 `route_ids`、状态目标对 `pairs` 和 `prepared_file`。缓存内含输入 token、assistant 监督标签与逐轮地图快照。小例中 `board_row=null` 只表示构造示例，不能直接作为正式采样行输入训练。

当前 50k 与 10k 不是不同环境；10k 从相同来源按棋盘与路径长度抽取。地图标签的权重版本必须与该批数据的实际来源一致。原始数据、token 缓存与全部轨迹保留在 Git 外。正式条件见 `experiments/map_sft_baselines/configs/`。
