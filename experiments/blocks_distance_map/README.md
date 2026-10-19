# 积木状态地图

本目录维护积木状态编码器 Q、精确关系标签、地图监督与纯地图评测。任务规则和提示见[任务入口](../../tasks/blocks/README.md)，共享方法见[QMAP_METHOD](../../tasks/QMAP_METHOD.md)。

## 当前入口

正式任务为原八形状、12次构造积木清空。所选模型和来源以 [selection.json](../../tasks/blocks/Qmap/selection.json) 为准；成绩、未达要求及关键比较见[12块报告](results/blocks12_joint_report.md)。验收定义只维护在[清空验收配置](configs/blocks12_clear_acceptance.json)。

| 内容 | 入口 |
|---|---|
| 已采用模型的训练配置 | [blocks12_decision_clear.json](configs/blocks12_decision_clear.json) |
| 最近原采样续训对照配置 | [blocks12_original_continuation.json](configs/blocks12_original_continuation.json) |
| 完整状态图与清空距离索引 | [full_graph_data.py](src/full_graph_data.py)、[symbolic_graph.py](src/symbolic_graph.py)、[clear_distance_index.py](src/clear_distance_index.py) |
| 联合训练与采样 | [full_graph_run.py](src/full_graph_run.py)、[full_graph_sampling.py](src/full_graph_sampling.py)、[broad_sampling.py](src/broad_sampling.py) |
| 固定题集评测 | [blocks12_eval.py](src/blocks12_eval.py)、[1000题](../../tasks/blocks/data/clear1000.json) |
| 本地运行定位 | [runtime 索引](../../runs/qmap_dual_20261018/current_qmap_runtime.json)（Git 外；运行时再核对） |

配置描述已经定义的条件，不表示任务仍在运行或可以自动续训。运行采用的配置应与对应产物中的 `config.json` 和源码版本一起解释。

## 可复用接口

| 模块 | 职责 |
|---|---|
| [model.py](src/model.py) | 目标无关的棋盘编码、共享 StateEncoder、固定距离和排序损失 |
| [oracle.py](src/oracle.py) | 差集精确覆盖；有限距离、已证明不可达与预算不足严格分开 |
| [multiboard_data.py](src/multiboard_data.py) | 多初盘、合法轨迹、状态与目标采样 |
| [full_graph_eval.py](src/full_graph_eval.py) | 完整图上的关系与决策诊断 |
| [failure_diagnosis.py](src/failure_diagnosis.py) | 同题新旧模型首次错误分叉与得失分析 |
| [FrozenBoardMap](../flamingo_map_reader/src/blocks.py) | 供 LLM 读取接口加载冻结地图 |

精确 oracle 用于标签和事后分析；纯地图求解只按合法后继的 Q 距离贪心选择。若使用候选过滤或搜索，须作为独立条件。固定监测集已参与选模，不作为独立泛化证据。

## 运行方式

以下是参数形式；实际 CONFIG、RUN、CHECKPOINT 和预算来自本次已授权条件。续训需要完整优化器和各 rank 采样状态。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m experiments.blocks_distance_map.src.full_graph_run \
  --config CONFIG --out RUN --resume CHECKPOINT --source-commit COMMIT
python -m experiments.blocks_distance_map.src.blocks12_eval \
  --config experiments/blocks_distance_map/configs/blocks12_clear_acceptance.json \
  --checkpoint CHECKPOINT --out EVALUATION --source-commit COMMIT --device cpu
```

机器、环境与传输方法见[开发机手册](../../DEVELOPMENT_MACHINES.md)。数据、断点、逐题结果及曝光记录留在 Git 外运行目录；README 不保存启动时间线或另抄采样比例。

## 独立最终评测

独立最终评测的准备入口为 [blocks12_sealed.py](src/blocks12_sealed.py)，参数见[配置](configs/blocks12_sealed_final.json)。`prepare` 必须显式提供 `--excluded-training-roots` 和 `--excluded-monitoring-cases`（各可传多个文件，监测集包含 `clear1000.json`），只生成冻结题集，不读取模型。确定候选后，`final-evaluate` 再显式指定 `--checkpoint` 与 `--parent-checkpoint`。它排除所提供的训练／监测初盘，不保证中间状态全部未见；非空目标仅作辅助。代码已做本地检查，尚无该入口的最终能力结果。

## 历史结果与复现

- [多棋盘与树关系结果](results/report.md)：保留旧地图的研究结论；历史 SFT 使用的地图来源也见任务 selection。
- [早期过程归档](results/archive/qmap_progression.md)：仅供追溯，不是当前任务清单。
- 早期单棋盘、多目标与树关系条件保留在 `configs/`，对应 `run.py`、`multiboard_run.py`、`tree_supervision_run.py`、`tree_continue_run.py`。复现时按原配置与源码版本运行，不把它们叠加为当前训练要求。

## Q 坐标与转移图可视化

[图例与解释边界](results/qmap_geometry.md)区分 Q 投影、真实转移和棋盘示意。数据由 [export_geometry.py](src/export_geometry.py) 导出，[plot_geometry.py](results/plot_geometry.py) 重生成报告图；绘图不更新模型。
