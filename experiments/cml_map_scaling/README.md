# Step 1：固定图的 Q/V 认知地图与规模检验

按用户附带的最新四步 TODO，先独立验证认知地图，不接入 LLM，不开展积木接口训练。每张固定图各自学习 Q/V；同一图上改变起终点不重新训练。改变边或节点组合属于另一张地图，不作跨图迁移主张。项目顺序与完成状态只维护在[当前状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

## 本次重跑：探索维度与更新方法

用户明确将探索优先限定于当前 Step 1：先学好 `Q o_{t+1} ≈ Q o_t + V a_t`，维度与更新方法作为可调选择，按实际效果推进。采用 [exploration.json](configs/exploration.json) 进行小型对照；沿用首轮所有图、探索轨迹、重放顺序和训练轮数。相同维度的两种方法使用相同初始化。

主看转移误差，以及预测向量最近的节点是否为正确后继；同时记录状态分散程度。远近关系用于描述地图结构，简单余弦选择器作为参考。完整梯度条件对每次转移的平方误差更新前态 Q、后态 Q 和动作 V，使用与反向传播等价的解析梯度；各坐标有效步长保持与局部条件一致。首轮结果保留，新运行单独保存。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.explore --output runs/cml_step1_exploration
```

## 首轮问题与判据

只看一步转移的局部学习，能否让图上距离较远的节点在表示空间中也较远？主要比较同一随机初始化在学习前后，所有不同节点对的图最短距离与原高维欧氏距离的 Spearman 秩相关，并按图距离报告欧氏距离均值和标准差。平方根曲线只是描述性拟合，不以拟合一条曲线证明机制。

补充测量用动作向量与目标差的余弦相似度选择真实合法动作，逐步到达目标；全部不同起终点对均纳入，显式检查终点，失败与成功路径的长度分别报告。这是 Q/V 地图的几何选择器诊断；不等同于带 W/G、动作历史抑制、噪声或奖励筛选的完整 GCML。训练没有起终点任务、专家动作或最短路标签；评测状态采用真实后继节点的 Q 表示，因此也不是纯内部状态累加想象。

零转移误差本身不足以证明远近性质：任意固定节点向量都可通过边专属 V 拟合准确后继。这一反例在正确性测试中明确覆盖。随机初始化对照用于判断本次实际学习过程是否改善几何。

## 算法来源与实现边界

- [CML 2024 正文](https://www.nature.com/articles/s41467-024-46586-0)，式 2–3：同一个更新前误差用于修改后继节点 Q 和动作 V，当前节点 Q 在该次更新中保持不变。
- [CML 官方 notebook](https://github.com/IGITUGraz/Cognitive-Map-Learner/blob/e36a17771bbe758c1d6b5f0446aa4a167b154a12/Cognitive%20Map%20Learner.ipynb)：参考初始化、学习率与小图规模；其图示使用轨迹级快照更新。
- [GCML 官方 notebook](https://github.com/LH-cbicr/GCML/blob/ff76859b71a2bc2056b50f5e052475351c007f76/gcml_abstract_graph.ipynb)：复用固定原图入口、动作编号与随机图生成程序。

本实现采用论文的**逐转移顺序更新**，明确处理一条轨迹重复到达同一节点的多次更新。两份 notebook 均使用重复索引的批量原地赋值；它不是顺序更新，重复索引的累积语义也不明确。GCML notebook 又以大 batch 加载后仅处理 `trajectory[0]`，CML 原 notebook 则以 batch size 1 遍历。这些区别使本研究只能称作“局部规则机制复现”，不能称为发布 notebook 或 Fig. 3 数值的逐项复现。

只学习 Q/V：原算法中的 W/G 不反馈到这两个矩阵，因此不需要为距离指标训练 W/G。没有显式归一化、正则化、监督距离或自动增加状态维度。

随机图保持发布生成程序的逻辑；其 `min_edges/max_edges` 参数**不是对称化后的严格度数上下界**，结果中报告实际度数。固定原图单列，随机图规模比较独立训练；不按结果筛图。较大图按节点数同比增加探索轨迹，固定每条轨迹的长度与重放轮数，以免把固定总探索预算的不足误当作容量失败；实际动作覆盖率仍需测量。该比较因此不是固定计算量的比较。

## 运行与产物

首轮正式合同为 [configs/step1.json](configs/step1.json)，不在说明文字中复制参数。代码只需 NumPy；顺序更新适合小型 CPU 数组，使用开发机 CPU，避免逐转移 GPU kernel 调度开销。CPU/GPU 不是待比较条件。

```bash
python -m unittest discover -s experiments/cml_map_scaling/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.run --output runs/cml_step1 --group official
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.run --output runs/cml_step1 --group scaling
```

先用固定原图测量成本，再运行已经固定的规模合同。每张图目录记录准确邻接、动作、随机轨迹、初始及最终 Q/V、距离对、规划策略与每个起终点的结果、训练曲线、覆盖率和耗时。原始产物放忽略目录；正式运行前提交源码和配置。输出目录拒绝覆盖已有图结果，不自动重试失败样本。

## 当前结论

重跑的四个条件、共 64 张地图已全部完成，全部有向动作的后继识别均为 100%。128 维、1000 维、2048 维及完整梯度更新都能学好转移；512 节点的距离相关分别为 0.559、0.585、0.590 和 0.541。完整结果见[报告](results/report.md)，下一步讨论见 [STEP2_DISCUSSION.md](STEP2_DISCUSSION.md)。

导出重跑的逐图数据、分组摘要与远近关系图（绘图依赖 Matplotlib）：

```bash
python -m experiments.cml_map_scaling.src.explore_report --run-dir runs/cml_step1_exploration --output experiments/cml_map_scaling/results --figures
```

首轮图表的再生成方式：

（额外依赖 Matplotlib 3.10.6）：

```bash
python -m experiments.cml_map_scaling.src.report --run-dir runs/cml_step1 --output experiments/cml_map_scaling/results
```

## Roadmap 可视化

原始 32 节点图及 128／256 节点 seed 0 case 的 t-SNE 图见[结果报告](results/report.md)。绘图合同为 [configs/roadmap_visualization.json](configs/roadmap_visualization.json)，保留全部真实连边，使用固定参数与种子。除已有绘图依赖外，需要 scikit-learn 1.7.2；本次运行使用 SciPy 1.16.2。

```bash
python -m experiments.cml_map_scaling.src.roadmap --input-root runs/cml_step1_exploration/local1000 --output experiments/cml_map_scaling/results
```

输出独立单图、规模并排图和各规模“原图随机布局—学习地图 t-SNE”对照图的 PNG／SVG／PDF，以及每张图的节点坐标、真实边表和投影指标。追加 `--pairs-only` 可只更新原图／学习地图对照。

### 二维长边与延长训练的比较

128／256 节点 seed 0 图中，二维最长的 5 条单步边在高维中仅为平均单步边长的 1.02–1.16 倍；从 20 轮续训到 100 轮，距离秩相关基本不变。这两个 case 的结果支持二维投影拉伸的解释。图、逐边数据和各轮比较已加入[报告](results/report.md)。合同见 [long_edge_diagnostic.json](configs/long_edge_diagnostic.json)。

以下输入目录应包含局部 1000 维条件的两个 case，每个 case 含 20 轮 `map.npz` 及其 `inputs.npz`；续训使用原数据和连续的重放随机序列。第二条命令仅导出结果，复用已有二维坐标，不重新拟合 t-SNE。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.long_edges --input-root runs/cml_step1_exploration/local1000 --run-dir runs/cml_long_edges_100
python -m experiments.cml_map_scaling.src.long_edges --input-root runs/cml_step1_exploration/local1000 --run-dir runs/cml_long_edges_100 --output experiments/cml_map_scaling/results
```
