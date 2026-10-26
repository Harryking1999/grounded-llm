# 实验与实现索引

从[任务工作区](../tasks/README.md)进入具体任务；研究优先级只见[状态页](../docs/PROJECT_STATUS_AND_TODO.md)。本页说明代码归属与复用关系。

## 当前任务入口

| 工作 | 实现与配置 | 结果 |
|---|---|---|
| 积木 Qmap | [blocks_distance_map](blocks_distance_map/README.md) | [12块清空](blocks_distance_map/results/blocks12_joint_report.md) |
| 寻路 Qmap | [path_qmap](path_qmap/README.md) | [报告](path_qmap/results/report.md) |
| 汉诺塔 Qmap | [hanoi_qmap](hanoi_qmap/README.md) | [报告](hanoi_qmap/results/report.md) |
| 二阶魔方 Qmap | [cube2_qmap](cube2_qmap/README.md) | [报告](cube2_qmap/results/report.md) |
| 8-Puzzle Qmap | [puzzle8_qmap](puzzle8_qmap/README.md) | [报告](puzzle8_qmap/results/report.md) |
| 推箱子／移车出库／棋子交换Qmap | [new_tasks_qmap](new_tasks_qmap/README.md) | [报告](new_tasks_qmap/results/report.md) |
| 地图接入 LLM 与同数据 SFT | [map_sft_baselines](map_sft_baselines/README.md) | [主比较](map_sft_baselines/results/report.md)、[小数据诊断](map_sft_baselines/results/b2_four_controls.md) |
| 五任务纯文字基线 | [task_text_baselines](task_text_baselines/README.md) | [模型与预算比较](task_text_baselines/results/report.md) |

## 先复用这些实现

| 能力 | 现有入口 |
|---|---|
| Q 编码器、固定距离与 soft-rank | [core/qmap/model.py](../grounded_llm/core/qmap/model.py) |
| 共用采样、基础指标与冻结推理 | [core/qmap](../grounded_llm/core/qmap) |
| 积木规则、精确标签、提示与裁判 | [tasks/blocks](../grounded_llm/tasks/blocks) |
| 地图槽位、投影、cross-attention 与 reader 构造 | [core/reader](../grounded_llm/core/reader) |
| SFT 编码、模型、数据与训练；原子文件 | [core/sft](../grounded_llm/core/sft)、[files.py](../grounded_llm/core/files.py) |
| 多基线训练、评测与跨机传输 | [map_sft_baselines](map_sft_baselines/README.md#运行入口) |
| 寻路环境与距离标签 | [tasks/path](../grounded_llm/tasks/path)；既有题集入口 [prepare_path_suite.py](qwen_path_blocks/src/prepare_path_suite.py) |

任务训练独立编排，具体适配见 [QMAP_METHOD](../tasks/QMAP_METHOD.md)，层次见[根目录说明](../README.md#代码依赖与接入)。五类新任务共用[完整目标训练入口](new_tasks_qmap/README.md)，各任务维护精确数据适配；过桥已接入真实边成本，24点使用精确有理数DAG。五人／六人过桥、两棵树棋子交换及有限21／24已采用，移车限定选用，推箱子搁置；能力边界见任务总览；首轮配置归档于 `new_tasks_qmap/archive/configs/pilot/`。

SFT任务层提供 `Demonstration`，每轮含文字、答案和地图step；`encode_trajectory`生成监督token，`collate_examples`组批，公共 `train_update`按监督token加权更新。正式入口为 `train(config, baseline, model_path, data, out, pad_token_id=...)`；任务变换由 `PreparedRoutes(transform=...)`接入。已有[共享管线检查](../tests/test_shared_pipeline.py)只证明接口接通。

## 历史实验与仍在使用的依赖

| 目录 | 保留原因 |
|---|---|
| [flamingo_map_reader](flamingo_map_reader/README.md) | 历史冻结读取接口研究与诊断；共享 reader 已迁入 core，旧路径兼容缓存和复现 |
| [llm_task_sft](llm_task_sft/README.md) | 早期无地图 SFT 和传输；动作协议与学习率基础已迁出 |
| [external_map_interface](external_map_interface/README.md)、[cml_map_scaling](cml_map_scaling/README.md) | 旧 Q/V 与连续 token 研究；共享寻路规则已迁入任务层 |
| [puzzle_qmap](puzzle_qmap/README.md) | 旧公共训练与导入兼容；当前汉诺塔、魔方用各自入口 |
| [gcml_counterexamples](gcml_counterexamples/README.md)、[qwen_path_blocks](qwen_path_blocks/README.md)、[sol_dag_blocks](sol_dag_blocks/README.md) | 历史模型基线；规则、题集生成仍有调用 |
| [state_interface_pilot](state_interface_pilot/README.md) | 早期积木连续 token 读出及公共 harness 使用示例 |
| [gcml_direct_api_blocks](gcml_direct_api_blocks/README.md)、[gcml_direct_api_path32](gcml_direct_api_path32/README.md) | 早期直接 API 协议与冻结题集 |

实验 README 维护目的和入口，`configs/` 维护参数，`src/` 维护实验编排及专属诊断，`tests/` 维护对应风险的检查，`results/` 保存研究结论和必要证据。历史实验不默认重跑；兼容转发不保存第二套实现。
