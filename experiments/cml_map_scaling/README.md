# Step 1：固定图的 Q/V 认知地图与规模检验

按用户附带的最新四步 TODO，先独立验证认知地图，不接入 LLM，不开展积木接口训练。每张固定图各自学习 Q/V；同一图上改变起终点不重新训练。改变边或节点组合属于另一张地图，不作跨图迁移主张。项目顺序与完成状态只维护在[当前状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

## 问题与判据

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

唯一正式合同为 [configs/step1.json](configs/step1.json)，不在说明文字中复制参数。代码只需 NumPy；顺序更新适合小型 CPU 数组，使用开发机 CPU，避免逐转移 GPU kernel 调度开销。CPU/GPU 不是待比较条件。

```bash
python -m unittest discover -s experiments/cml_map_scaling/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.run --output runs/cml_step1 --group official
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.cml_map_scaling.src.run --output runs/cml_step1 --group scaling
```

先用固定原图测量成本，再运行已经固定的规模合同。每张图目录记录准确邻接、动作、随机轨迹、初始及最终 Q/V、距离对、规划策略与每个起终点的结果、训练曲线、覆盖率和耗时。原始产物放忽略目录；正式运行前提交源码和配置。输出目录拒绝覆盖已有图结果，不自动重试失败样本。

## 当前结论

已完成全部 16 张地图。原始图的距离秩相关从 0.082 升至 0.941；随机图的该指标随规模从 0.842 降至 0.585，简单余弦选择器的到达率从 100% 降至 51.36%。全部动作已探索、转移拟合误差很低，仍未保证可靠规划。完整结论、统计口径及限制见[结果报告](results/report.md)。该阶段不能证明 LLM 状态使用改善，也不外推到共享跨图模型。

生成报告与图表（额外依赖 Matplotlib 3.10.6）：

```bash
python -m experiments.cml_map_scaling.src.report --run-dir runs/cml_step1 --output experiments/cml_map_scaling/results
```
