# 二阶魔方 Qmap

研究问题：在完整 2×2×2 魔方群上离线生成精确关系后，共享状态编码器能否在未参与任何训练损失的状态上表达远近、正确排序真实合法后继并支持纯 Q 贪心到达？正式参数见 [configs/study.json](configs/study.json)，共同模型与指标定义见 [共享核心](../puzzle_qmap/README.md)，结果见 [results/report.md](results/report.md)。

针对旧数据的覆盖缺口，新一轮使用 [breadth_depth.json](configs/breadth_depth.json)，第一性原理推导、覆盖审计、深路径与完整分叉、评测边界见 [DATA_DESIGN.md](DATA_DESIGN.md)。旧 `study.json` 保留为已有纯 Q 结果的合同，不覆盖旧产物。

用户确认后的分层均衡续训使用 [balanced_epochs.json](configs/balanced_epochs.json)：固定复用广深数据及纯 Q／Adam，距离对按真距离、排序组按父状态真距离分层；每批等额配给，余数轮换，每层独立无放回遍历后重排。完整候选软目标与两个损失的权重均保持原定义。学习率逐渐降低，执行完整提高后的预算；各层实际覆盖轮次单独记录。起始权重参与验证选择，另存末步权重，避免覆盖门槛排除更好的早期模型。正式预算、最低覆盖与学习率只维护在配置中。

用户已授权六动作纯 Q 对照，合同为 [quarter_turn_local.json](configs/quarter_turn_local.json)，具体数据与比较边界见 [DATA_DESIGN.md](DATA_DESIGN.md#六动作局部图与在线呈现)。`cube2_qtm` 只保留 U/U'、R/R'、F/F'；沿用同一个共享编码器、对称距离、Huber 和完整软排序，从头训练，不添加转移预测头。六动作与九动作的 oracle、标签、checkpoint 和输出目录分别保存。

```bash
python -m experiments.puzzle_qmap.src.run --config experiments/cube2_qmap/configs/quarter_turn_local.json --out runs/cube2_quarter_turn_RUN --source-commit COMMIT --device cuda:1
python -m experiments.cube2_qmap.src.compare --run runs/cube2_quarter_turn_RUN --reference-run runs/cube2_balanced_epochs_90ec4ca --device cuda:1
```

相对根状态按真实距离均衡、层内无放回遍历，每次抽取重新为整个局部图选择共同左乘，所有图节点与目标均避开保护的评测端点。`data.npz` 中 `pairs_train`／`tasks_train` 在此条件下只是固定训练监测集；训练样本来自在线数据流。各根层覆盖、实际距离对／排序父层分布分别记录，不能将根层均衡说成展开后每层严格等额。固定训练监测集与验证集记录 Huber、软排序 CE 及目标熵；最佳模型按验证目标选择，测试只在训练结束后使用。

六动作首轮已经完成，完整 loss 曲线及单步／循环分析见 [结果报告](results/report.md)、[紧凑证据](results/quarter_turn_summary.json)。用户随后要求针对错误排序造成的往返循环修复，匹配续训合同为 [quarter_turn_progress.json](configs/quarter_turn_progress.json)。两组从同一个六动作 Q／Adam 开始，共用在线流和验证选择；对照继续绝对距离 Huber，干预将它替换为所有真实动作边的距离差 Huber。完整软排序及 Q 参数结构均不变。两组共同按验证最短动作率最大值选模型，平局保留较早 checkpoint，初始权重可获选；末步另报，不按测试结果改选。

距离差替换的匹配结果没有修复长程失败，随后按用户的针对性修复要求检验 `progress_margin` 条件：恢复绝对距离 Huber，保留完整软排序，再监督“至少一个真实最优后继比当前状态更近，并领先最佳错误候选”。这是新增的第三项损失，具体定义及失败动机见数据设计；模型参数和推理器仍不变。该条件从同一初始权重与 Adam 开始，复用原匹配对照，不沿测试表现选择起点。

```bash
python -m experiments.cube2_qmap.src.progress_experiment --contract experiments/cube2_qmap/configs/quarter_turn_progress.json --condition edge_difference --out runs/cube2_progress_RUN --source-commit COMMIT --device cuda:1
python -m experiments.cube2_qmap.src.decision_audit --run runs/cube2_progress_RUN --device cuda:1
```

```bash
python -m experiments.puzzle_qmap.src.run --config experiments/cube2_qmap/configs/balanced_epochs.json --out runs/cube2_qmap/BALANCED_RUN --source-commit COMMIT --device cuda:1 --prepared-run runs/cube2_qmap/breadth_depth --checkpoint runs/cube2_qmap/breadth_depth/best.pt
```

`--prepared-run` 复用相同的数据、oracle 与划分；`--checkpoint` 恢复 Q 和 Adam，并使用新的固定种子采样流。这不声称恢复父运行的逐步随机序列。每次验证记录固定训练监测集、分层验证动作率与学习率；测试仅在模型按验证目标选定后评估。

## 状态与动作约定

采用 [Kociemba 的角块顺序与朝向定义](https://www.kociemba.org/math/cubielevel.htm)：URF/UFL/ULB/UBR/DFR/DLF/DBL/DRB。状态是八角块排列 `cp` 和八个朝向 `co`；朝向取 0/1/2，总和必须模 3 为零。二阶魔方没有棱块，**不施加三阶角块／棱块奇偶耦合限制**。

整块旋转作为参考系等价：输入先置于 DBL 角块回到 DBL 且朝向为零的规范框架。其余七块任意排列、六个独立朝向，得到 `7! × 3^6 = 3,674,160` 个状态。编号是七块排列的 Lehmer rank 与六位三进制朝向的积编码；`state_id` 拒绝非法排列、非法扭转和未规范化朝向。当前 API 显式接收规范化角块，不含拍照识别、贴纸解析或自动转动实体魔方的接口。

动作固定为 U/U2/U'、R/R2/R'、F/F2/F'，从该面外侧正视，正转为顺时针。每次转面，无论 90/180/270 度都计一步（HTM）。这些动作均保持 DBL 基准；每个合法状态有九个合法后继。D/L/B 和整块旋转不作为本动作空间的输入，不能把未规范化六面动作轨迹直接当作本数据。转动定义参照 [作者给出的 move tables](https://github.com/hkociemba/RubiksCube-TwophaseSolver/blob/master/cubie.py)，实现仅使用角块规则，测试另用三维贴纸旋转独立核对。

## 完整 oracle 与数据

转移表按角块排列和朝向分解，不储存巨大的完整邻接矩阵。从复原态完整 BFS，得到覆盖全部状态、最大距离为 11 的 uint8 表。任意两状态的精确距离通过群相对状态 `source^-1 * goal` 查表；`solve` 选真实距离每步减一的后继，因此支持任意规范目标，不只复原态。

原 `study.json` 采样时均匀选择真实步数，再从对应 BFS 深度层取相对状态，与随机起点合成目标；不是把打乱动作数误当成最短距离。原合同按完整状态空间的稳定哈希划分训练／验证／测试。对训练状态对及整组候选作共同左乘增强 `s -> k*s`，保持精确距离与动作后继；增强后全部训练状态、目标与候选仍必须属于训练划分。新 `breadth_depth.json` 则穷举非恒等相对决策类型，保护新旧评测的实际状态端点，并增加同目标路径和闭合分叉；划分条件不同，完整边界见数据设计。每个排序训练组保留全部九个合法候选。距离表覆盖测试空间是**离线裁判**，不作为 Q 输入。编码器和完整软排序沿用积木实现，只有 Q 可训练。

Q 输入为每个位置的 `(角块身份, 朝向)` 联合 one-hot，保留朝向与身份对应关系。全空间合法和非法状态检查与学习分开。训练若只得到局部远近而连续贪心失败，应报告该失败，不将精确 solver 的成功当成 Q 已学会求解。

## 运行

```bash
python -m unittest discover -s experiments/cube2_qmap/tests -v
python -m unittest discover -s experiments/puzzle_qmap/tests -v
python -m experiments.puzzle_qmap.src.run --config experiments/cube2_qmap/configs/study.json --out runs/cube2_qmap/RUN_ID --source-commit COMMIT --device cuda:1
python -m experiments.puzzle_qmap.src.solve --task cube2 --scramble "R U F2 R'" --oracle runs/cube2_qmap/RUN_ID/oracle_depth.npy
python -m experiments.puzzle_qmap.src.solve --task cube2 --scramble "R U F2 R'" --checkpoint runs/cube2_qmap/RUN_ID/best.pt
```

完整 BFS 覆盖检查只需数秒，最终 oracle 文件约 3.7 MB。测试同时检查全部深度计数、独立几何转动、逆动作、编码往返、无边奇偶约束、非法输入、任意目标短程 BFS 与最短解重放。冻结 Q 推理不加载 oracle，输出动作及状态轨迹方便逐步检查。
