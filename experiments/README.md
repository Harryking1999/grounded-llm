# 实验与实现索引

从[任务工作区](../tasks/README.md)进入具体任务；研究优先级只见[状态页](../docs/PROJECT_STATUS_AND_TODO.md)。本页说明代码归属与复用关系，不授权启动实验。

## 当前任务入口

| 工作 | 实现与配置 | 结果 |
|---|---|---|
| 积木 Qmap | [blocks_distance_map](blocks_distance_map/README.md) | [12块清空](blocks_distance_map/results/blocks12_joint_report.md) |
| 寻路 Qmap | [path_qmap](path_qmap/README.md) | [报告](path_qmap/results/report.md) |
| 汉诺塔 Qmap | [hanoi_qmap](hanoi_qmap/README.md) | [报告](hanoi_qmap/results/report.md) |
| 二阶魔方 Qmap | [cube2_qmap](cube2_qmap/README.md) | [报告](cube2_qmap/results/report.md) |
| 8-Puzzle Qmap | [puzzle8_qmap](puzzle8_qmap/README.md) | [报告](puzzle8_qmap/results/report.md) |
| 地图接入 LLM 与同数据 SFT | [map_sft_baselines](map_sft_baselines/README.md) | [主比较](map_sft_baselines/results/report.md)、[小数据诊断](map_sft_baselines/results/b2_four_controls.md) |
| 五任务纯文字基线 | [task_text_baselines](task_text_baselines/README.md) | [模型与预算比较](task_text_baselines/results/report.md) |

## 先复用这些实现

| 能力 | 现有入口 |
|---|---|
| Q 编码器、固定距离与 soft-rank | [blocks_distance_map/src/model.py](blocks_distance_map/src/model.py) |
| 共用采样与基础指标 | [qmap_sampling.py](../grounded_llm/qmap_sampling.py)、[qmap_metrics.py](../grounded_llm/qmap_metrics.py) |
| 积木规则与精确裁判 | [gcml_counterexamples/src/blocks.py](gcml_counterexamples/src/blocks.py)、[blocks_distance_map/src/oracle.py](blocks_distance_map/src/oracle.py) |
| 地图槽位、投影、cross-attention 与 reader 构造 | [state_q.py](flamingo_map_reader/src/state_q.py)、[memory.py](flamingo_map_reader/src/memory.py)、[fusion.py](flamingo_map_reader/src/fusion.py)、[train.py](flamingo_map_reader/src/train.py) |
| 积木动作解析、事实依据、原子文件与传输基础 | [protocol.py](llm_task_sft/src/protocol.py)、[files.py](llm_task_sft/src/files.py)、[bridge.py](llm_task_sft/src/bridge.py) |
| 多基线训练、评测与跨机传输 | [map_sft_baselines](map_sft_baselines/README.md#运行入口) |
| 寻路环境、距离标签和题集 | [transitions.py](external_map_interface/src/transitions.py)、[core.py](cml_map_scaling/src/core.py)、[prepare_path_suite.py](qwen_path_blocks/src/prepare_path_suite.py) |

任务训练独立编排，具体适配见 [QMAP_METHOD](../tasks/QMAP_METHOD.md)。复用组件不等于复用其旧实验条件。

## 历史实验与仍在使用的依赖

| 目录 | 保留原因 |
|---|---|
| [flamingo_map_reader](flamingo_map_reader/README.md) | 已完成的冻结读取接口研究；reader、地图槽位和评分仍被当前 SFT 使用 |
| [llm_task_sft](llm_task_sft/README.md) | 早期无地图 SFT；动作协议、调度基础仍有调用 |
| [external_map_interface](external_map_interface/README.md)、[cml_map_scaling](cml_map_scaling/README.md) | 旧 Q/V 与连续 token 研究；寻路环境及最短路组件仍被使用 |
| [puzzle_qmap](puzzle_qmap/README.md) | 旧公共训练与导入兼容；当前汉诺塔、魔方用各自入口 |
| [gcml_counterexamples](gcml_counterexamples/README.md)、[qwen_path_blocks](qwen_path_blocks/README.md)、[sol_dag_blocks](sol_dag_blocks/README.md) | 历史模型基线；规则、题集生成仍有调用 |
| [state_interface_pilot](state_interface_pilot/README.md) | 早期积木连续 token 读出及公共 harness 使用示例 |
| [gcml_direct_api_blocks](gcml_direct_api_blocks/README.md)、[gcml_direct_api_path32](gcml_direct_api_path32/README.md) | 早期直接 API 协议与冻结题集 |

实验 README 维护目的和入口，`configs/` 维护参数，`src/` 维护实现，`tests/` 维护对应风险的检查，`results/` 保存研究结论和必要证据。历史实验不默认重跑，仍被导入的源码不随文档归档移动。
