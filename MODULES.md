# 模块入口

当前研究的结论与下一步见[项目状态页](docs/PROJECT_STATUS_AND_TODO.md)；正式参数以各实验的 `configs/` 为准。本页只说明代码职责，不复制运行合同。

| 研究 | 核心模块 | 职责 |
| --- | --- | --- |
| [积木共享 Q-map](experiments/blocks_distance_map/README.md) | `oracle.py`、`model.py` | 精确有向距离标签、棋盘编码器与 Q 距离 |
| 积木共享 Q-map | `multiboard_data.py`、`tree_supervision_data.py`、`tree_continue_data.py` | 棋盘划分、状态对及树关系采样 |
| 积木共享 Q-map | `multiboard_run.py`、`tree_supervision_run.py`、`tree_continue_run.py` | 基础训练、树关系训练和清空／长程续训 |
| 积木共享 Q-map | `rollout_eval.py`、`tree_breadth_eval.py`、`tree_clear_eval.py`、`tree_length_eval.py` | 合法后继选择与封存棋盘分层评测；棋盘划分核对共用 `fresh_official_boards` |
| [寻路外部地图](experiments/external_map_interface/README.md) | `q_map.py`、`transitions.py` | 读取学到的 Q/V、执行合法动作并给出候选距离 |
| 寻路外部地图 | `evaluate_path256_continuous.py` | 完整路线基线与逐节点地图插入协议 |
| 寻路外部地图 | `evaluate_path256_continuous_batch.py`、`summarize_path256_continuous_batch.py` | 冻结五图批量运行与独立重放汇总 |

`grounded_llm/` 是早期连续 state token 实验的共享实现，现不参与上述两条主线。`grounded_llm/graph_data.py`、`graph_prompts.py`、`graph_scoring.py` 保留为旧图 Step 2 的兼容导出；其实现分别在 `data.py`、`prompts.py`、`scoring.py`。旧实验入口见[连续 token 读出实验](experiments/state_interface_pilot/README.md)和[图 Step 2](experiments/cml_map_scaling/README.md)。
