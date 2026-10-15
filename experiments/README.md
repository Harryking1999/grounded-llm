# 实验目录

当前主实验：[同数据地图与 SFT 多基线](map_sft_baselines/README.md)。其余目录分别保存独立协议或历史证据，目录内的“当前”以该实验语境为准。

项目进度只维护在[状态页](../docs/PROJECT_STATUS_AND_TODO.md)。各实验有独立的参数、数据和比较边界；已完成的历史实验不作为新运行的默认启动指令。

| 目录 | 作用 |
|---|---|
| [map_sft_baselines](map_sft_baselines/README.md) | 当前同数据对照：地图、主干训练范围与接口容量 |
| [llm_task_sft](llm_task_sft/README.md) | 独立无地图任务 SFT 的历史协议与结果 |
| [flamingo_map_reader](flamingo_map_reader/README.md) | 已完成的冻结读取接口、两任务证据和 FFN 负结果 |
| [blocks_distance_map](blocks_distance_map/README.md) | 冻结积木 Q 的来源、编码器与精确裁判 |
| [hanoi_qmap](hanoi_qmap/README.md)、[cube2_qmap](cube2_qmap/README.md) | 三层汉诺塔／二阶魔方的规则、精确数据、Q 训练与独立求解 |
| [puzzle_qmap](puzzle_qmap/README.md) | 两项可逆新任务共享的距离、排序、转移训练及评测核心 |
| [external_map_interface](external_map_interface/README.md) | 寻路 Q/V 与环境依赖；显式距离接口的历史试验 |
| [cml_map_scaling](cml_map_scaling/README.md) | 图地图训练与几何结果；旧连续 token 接口 |
| [qwen_path_blocks](qwen_path_blocks/README.md)、[sol_dag_blocks](sol_dag_blocks/README.md)、[gcml_counterexamples](gcml_counterexamples/README.md) | 独立模型基线 |
| [state_interface_pilot](state_interface_pilot/README.md) | 旧积木连续 token 读出 |

每个研究用 `README.md` 说明目的和入口，`configs/` 保存正式合同，`src/` 保存可复用实现，`tests/` 保存针对性检查。原始数据、模型、回答和日志写入忽略的 `runs/` 等目录；Git 只收录源码、配置和与决策有关的结果摘要。历史及归档内容不维护当前进度。
