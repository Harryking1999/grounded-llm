# 实验目录

当前主线是通过 cross-attention 让冻结 LLM 读取并利用地图。先读[地图读取实验](flamingo_map_reader/README.md)及其[当前 design](flamingo_map_reader/DESIGN.md)；新架构试点见[本轮结果](flamingo_map_reader/results/report.md)，原两任务成绩见[上一轮报告](flamingo_map_reader/results/report_long_trajectory.md)，进度只维护在[项目状态页](../docs/PROJECT_STATUS_AND_TODO.md)。

| 目录 | 作用 |
|---|---|
| [flamingo_map_reader](flamingo_map_reader/README.md) | 当前读取接口：单图寻路与 1000 初始棋盘积木的完整轨迹 SFT |
| [blocks_distance_map](blocks_distance_map/README.md) | 冻结积木 Q 的来源、编码器与精确裁判 |
| [external_map_interface](external_map_interface/README.md) | 寻路 Q/V 与环境依赖；显式距离接口的历史试验 |
| [cml_map_scaling](cml_map_scaling/README.md) | 图地图训练与几何结果；旧连续 token 接口 |
| [qwen_path_blocks](qwen_path_blocks/README.md)、[sol_dag_blocks](sol_dag_blocks/README.md)、[gcml_counterexamples](gcml_counterexamples/README.md) | 独立模型基线 |
| [state_interface_pilot](state_interface_pilot/README.md) | 旧积木连续 token 读出 |

每个研究用 `README.md` 说明目的和入口，`configs/` 保存正式合同，`src/` 保存可复用实现，`tests/` 保存针对性检查。原始数据、模型、回答和日志写入忽略的 `runs/` 等目录；Git 只收录源码、配置和与决策有关的结果摘要。历史及归档内容不维护当前进度。
