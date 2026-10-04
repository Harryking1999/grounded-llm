# 已归档：地图读取接口的失败变体

**处理决定：下表每一行都不进入当前主证据链，不再续训，也不再据此安排新训练。**源码与配置保留，只为让已归档的运行可复现。当前有效证据见[主报告](../report.md)，当前设计见[实验设计](../../DESIGN.md)。

本轮只保留了其中一个结构：角色／编号做寻址 K、投影后的 Q 做状态 V（`memory_mode="address_key_state_value"`），门控可训练，`value_scale` 与 `fixed_gate_tanh` 都用默认值。本页只做索引与定位，不重新裁定已归档报告里的结论。

## 变体对照

| 变体 | 想解决什么 | 当时的读数 | 源码位置（按符号，不按行号） | 配置 |
| --- | --- | --- | --- | --- |
| 混合 K/V（`memory_mode="joint"`） | 冻结 LLM 通过一个可训练接口同时读地址与状态 | 训练首轮完整排序 16/16 拟合，留出 0/4（寻路与积木同） | `memory.py` 的 `MapMemoryEncoder`；`fusion.py` 取 `combine.out_features` 的混合分支；`train.py` 的 `mode` 默认值 | 不写 `memory_mode` 的旧配置（见下） |
| 固定门控 `tanh(g)=0.1` | 排除"门控关死、地图通路零输入" | 同图验证首轮最近候选 23/64；训练首轮也仅 101/256 | `fusion.py` 的 `fixed_gate_tanh`（含 `MapReader` 里冻结门控的分支）；`train.py` 的透传 | `pilot_path256_fixed_gate.json` |
| 关系 token 加权 `decision_focus_weight=16` | 加大排序与末尾 action 编号的损失权重 | 同图验证首轮最近候选 12/64 | `readout_aux.py`、`train.py`、`epoch_readout_queue.py` 里读取该键的位置 | `pilot_path256_relation_weighted.json` |
| 状态值尺度校准 `value_scale` | 对齐状态幅值与语言 hidden | 未单独裁定；随 swap 与十轮读出一起封存 | `fusion.py` 的 `value_scale`；`train.py` 的透传 | `path_readout_expanded.json`、`path_completion_ten_epochs*.json`（=14.41） |
| 自然 Q-only readout | 用"同状态不同目标"的物理合法 Q 对比做读出 | 未纳入本轮 | **未跟踪**：`src/prepare_natural_readout.py`、`tests/test_natural_readout.py` | **未跟踪**：`configs/path_natural_qonly_*.json` |

对应读数分别记在 [full_trajectory_history.md](full_trajectory_history.md)（16／4 小样本三条件与四图）、[swap_training.md](swap_training.md) 和 [readout_interface_diagnosis.md](readout_interface_diagnosis.md)（swap D/E/F、hidden-state 与探针）。

## 仍在源码里、但当前配置不用的东西

两个正式配置 `configs/path_single_long.json`、`configs/blocks1000_long.json` 都写 `"memory_mode": "address_key_state_value"`，且都不传 `fixed_gate_tanh` / `value_scale`。所以上表前四行在源码里是死路径：能加载、能复现旧运行，但不参与本轮训练。已加弃用标注的位置：

- `src/memory.py` — `MapMemoryEncoder` 类文档串
- `src/fusion.py` — 模块文档串，以及 `value_scale`、`fixed_gate_tanh` 两个参数
- `src/train.py` — `mode` 的 `"joint"` 默认分支，以及向 `MapReader` 透传两个废弃参数的调用

走 `"joint"` 这条废弃默认值的旧配置共 10 个：`kv_convergence.json`、`kv_convergence_extension.json`、`kv_pilot.json`、`kv_scale_followup.json`、`pilot_blocks.json`、`pilot_path256.json`、`pilot_path256_fixed_gate.json`、`pilot_path256_relation_weighted.json`、`path_readout_expanded_resume.json`、`path_readout_ten_epochs_extension.json`。

`configs/path_readout_gate.json` 名字里的 "gate" 指读出验收阈值（`minimum_*` 那组键），与注意力门控无关，不在上表内。
