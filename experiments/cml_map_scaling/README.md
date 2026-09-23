# Q/V 地图学习与历史 LLM token 接口方案

Step 1 的 Q/V 结果仍供当前显式 roadmap 路线复用。图 Step 2 属于 previous approach：将 Q 映射为冻结 LLM 的连续输入 token；已有 runner、正式配置、诊断和结果全部保留供复现，不是当前执行方案。当前方向见[研究简述](../../docs/RESEARCH_BRIEF.md)和[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

| 阅读入口 | 内容 |
| --- | --- |
| [Step 1 结果](results/report.md) | 转移、地图几何、规模与长边诊断 |
| [Step 2 runbook](STEP2_RUNBOOK.md) | 已封存的报告训练与一步动作评测方案 |
| [Step 2 配置](configs/step2.json) | 旧方案的模型、参数、数据划分、prompt 与 loss |
| [Step 2 验收](results/step2_acceptance.md) | 已完成的报告、一步选择、重编号与直接几何读出结论 |
| [原讨论稿](STEP2_DISCUSSION.md) | 保留会议想法和批注，供回溯参考 |

当前进度统一见[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

## Step 1 做了什么

从固定图的探索轨迹学习 Q/V，使当前状态加动作位移接近下一状态。每张图独立训练地图；评测一步转移误差、后继节点识别、图距离与表示距离的 Spearman 相关，另用简单余弦选择器测试到达率。

主比较为局部更新的不同维度，以及相同维度的完整梯度更新。条件共用图、探索数据、重放顺序和轮数；同维条件也共用初始化。较大图按节点数增加探索量，具体参数见 [exploration.json](configs/exploration.json) 及其继承的 [step1.json](configs/step1.json)。

已完成四个条件、64 张地图，所有后继识别均为 100%，并形成远近关系。图上每条边有专属动作向量，因此转移拟合本身不能证明几何或规划有效，这几项分别统计。

## 实现依据

- [CML 论文](https://www.nature.com/articles/s41467-024-46586-0)式 2–3：用更新前误差修改后继 Q 和动作 V，当前 Q 保持不变。
- [GCML 官方代码](https://github.com/LH-cbicr/GCML/tree/ff76859b71a2bc2056b50f5e052475351c007f76)：提供原图、动作编号、随机图生成和可视化参考。
- 本实现逐转移顺序更新；发布 notebook 的批量重复索引更新方式不同，因此本实验是局部规则机制复现。只学习 Q/V，未复现完整 GCML 的 W/G、噪声与轨迹筛选。
- 完整梯度条件对前态 Q、后态 Q 和 V 使用解析梯度。余弦选择器读取真实合法动作与真实后继 Q，无搜索或回溯。

## 运行与复现

以下为已有 Step 1 命令，在配置好 Python 环境后于仓库根目录运行。产物目录必须是新的目录。

```bash
python -m unittest discover -s experiments/cml_map_scaling/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.explore --output runs/cml_step1_exploration
python -m experiments.cml_map_scaling.src.explore_report --run-dir runs/cml_step1_exploration --output experiments/cml_map_scaling/results --figures
```

重建地图对照图（输入目录下包含三个 case 的 `map.npz` 与 `inputs.npz`）：

```bash
python -m experiments.cml_map_scaling.src.roadmap --input-root runs/cml_step1_exploration/local1000 --output experiments/cml_map_scaling/results
```

重建长边诊断：第一条从已有地图续训；已有续训产物时只需第二条导出结果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.long_edges --input-root runs/cml_step1_exploration/local1000 --run-dir runs/cml_long_edges_100
python -m experiments.cml_map_scaling.src.long_edges --input-root runs/cml_step1_exploration/local1000 --run-dir runs/cml_long_edges_100 --output experiments/cml_map_scaling/results
```

训练依赖 NumPy；绘图另需 Matplotlib，t-SNE 使用 scikit-learn。各次运行版本和来源见结果摘要。地图可视化与长边合同分别为 [roadmap_visualization.json](configs/roadmap_visualization.json)、[long_edge_diagnostic.json](configs/long_edge_diagnostic.json)。

## 结果保存

`results/` 保留一份报告、正文使用的 PNG、紧凑 CSV／JSON 和投影坐标。PDF／SVG 与其他重复图可由现有绘图脚本再导出，不进入 Git。原始轨迹、Q/V、训练快照与完整日志保存在 `runs/`。

旧图 Step 2 的训练入口为 `src/step2.py`，其余 `src/step2*.py` 与相应配置保存后续诊断。它们只用于复现 previous approach；新显式接口位于 [external_map_interface](../external_map_interface/README.md)。
