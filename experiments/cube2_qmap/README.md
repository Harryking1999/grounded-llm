# 二阶魔方 Qmap

**当前保留六动作纯 Q 基线，长程求解尚未成功。**整体模型先看[共享说明](../puzzle_qmap/README.md)，实测结论看[结果报告](results/report.md)。这里只解释任务和数据。

## 状态与动作

用八个角位置的角块身份 cp 与朝向 co 表示状态。角块身份为 0…7 的排列，朝向为 0/1/2，朝向和必须模 3 为零。采用 [Kociemba 角块约定](https://www.kociemba.org/math/cubielevel.htm)，不施加三阶魔方的角块／棱块奇偶耦合限制。

使用 DBL 角块固定且不扭转的规范参考系，共 7! × 3⁶ = **3,674,160** 个状态。接口要求输入已经处于此参考系，不包含贴纸识别或自动规范化。网络输入是“位置 × 角块身份 × 朝向”的联合独热，共 192 维；状态编号只是索引，不进入网络。

当前任务名 cube2_qtm，动作是 U、U′、R、R′、F、F′；从相应面外侧看，正转为顺时针 90°，每次动作计一步。每个状态均有六个合法后继；180° 需两步。图直径为 14，每条边到固定目标的真距离变化都为 ±1。环境仍可读取历史九动作 HTM 约定，旧权重与六动作权重不能混用。

## 训练数据

1. 从复原态完整 BFS。任意两状态的距离通过群相对状态查表，不能拿打乱序列长度作标签。
2. 每个相对根状态枚举全部六个单步和 36 个两步序列，合并为 34 个不同状态。起点及六个直接后继作为七个父节点，各自保留完整六候选。
3. 按根状态到目标的真距离分层，每层打乱、无放回遍历；抽到后将目标与整个局部图共同左乘同一个 k。此变换保持距离和动作边；对每个状态执行同一串右乘动作不能替代它。
4. 距离对包含节点到目标的长程关系和局部节点间的近程关系，均使用完整图最短距离。排序也使用全部真实候选，保留所有最优动作。
5. 训练图中所有输入排除保护的评测端点。data.npz 中的训练数组只是固定监测集，实际训练在线构造新呈现；根层均衡不等于展开后的所有距离层都严格均衡。

## 配置与运行

基线配置为 [quarter_turn_local.json](configs/quarter_turn_local.json)。清理后的默认配置可独立生成划分，无需旧运行目录；它不是已报告历史运行的原样复现，旧配置和数据在归档中。可选 reference_data 仅用于需要保持历史同题比较的情况。

```bash
python -m experiments.puzzle_qmap.src.run --config experiments/cube2_qmap/configs/quarter_turn_local.json --out runs/cube2_qmap/RUN_ID --source-commit COMMIT --device cuda:0
python -m experiments.puzzle_qmap.src.solve --task cube2_qtm --start 12345 --goal 0 --checkpoint runs/cube2_qmap/RUN_ID/best.pt
python -m unittest discover -s experiments/cube2_qmap/tests -v
```

运行目录必须不存在。纯 Q 求解不加载 oracle；精确求解是独立裁判条件。

## 固定度量对照

[metric_comparison.json](configs/metric_comparison.json) 定义 L1 与 L∞ 的同题、同初始权重对照；若 L∞ 未达到预先写定的验证到达条件，顺序运行 L2。复用归档基线的数据与 oracle，不续接其权重。编码器、输出范围、两项损失、在线采样流、步数和学习率计划相同。距离尺度沿用原 L1，L∞ 与 L2 使用原始范数；因此这是完整读出规则的比较，不能单独归因于范数形状。各自的初始化损失会不同。

```bash
python -u -m experiments.cube2_qmap.src.metric_study --contract experiments/cube2_qmap/configs/metric_comparison.json --prepared-run /PATH/TO/cube2_quarter_turn_8eed676 --out /PATH/TO/metric_study --source-commit COMMIT --devices cuda:0 cuda:1
```

程序只依据验证集决定是否运行 L2；测试结果不参与条件选择。最终同时报告验证选定与末步模型的距离误差、排序和完整到达，不把 loss 降低当作成功。运行、日志和完整结果集中在同一个 Git 外目录；启动与结论记录于结果报告。

L2 也不能精确表示四条边为 1、两条对角线为 2 的图四环：欧氏空间中距离 1+1=2 的等号迫使中间点位于两端中点，两条路径的中点因此重合。这不否定近似动作排序；对应检查见共享测试。结构调研与后续建议见[研究记录](results/structure_notes.md)。

## 逐项结构检查

[structure_checks.json](configs/structure_checks.json) 为小型多目标诊断合同，[structure_study.py](src/structure_study.py) 复用同一编码器与纯 Q 推理接口。先比较固定状态上的稀疏／完整距离监督及分组 L1-max；再在相同新呈现流上，分别加入边上界、路径进展或配对距离对称监督。各项不同时叠加，不加入目标条件读出或学习的转移头。

```bash
python -u -m experiments.cube2_qmap.src.structure_study --contract experiments/cube2_qmap/configs/structure_checks.json --prepared-run /PATH/TO/cube2_quarter_turn_8eed676 --out /PATH/TO/structure_checks --source-commit COMMIT --device cuda:2
```

完整距离与稀疏距离条件的状态编码和更新次数相同，标签曝光不同。固定呈现组按完整训练目标选模，仅用于有限集合拟合；新呈现组按验证目标选模。测试在选模后评估最佳与末步模型。原实验保护端点以及本次所有留出呈现中的输入状态均排除在训练之外；新呈现条件也不使用固定呈现的输入状态。其 `fit` 列在新呈现条件中是共同固定监测呈现，不能称为训练集成绩。动作候选和 rollout 中间状态由环境给出，且可能离开有限模板。

分组规则为固定块内绝对差相加、块间取最大；块宽保存在 checkpoint 的 `group_size`。构造性 L∞ 可行解只证明小集合可表示。留出呈现仍然复用同一组相对关系，不能当成完整魔方或未见关系类型泛化。

## 关系监督的使用效率

正式方向仍是通过距离与动作关系训练共享编码器，由网络自行组织 Q 坐标。直接坐标标签只用于 [coordinate_probe.json](configs/coordinate_probe.json) 与 [coordinate_shift_probe.json](configs/coordinate_shift_probe.json) 的有限拟合诊断；不是正式训练方案。[warm_start_checks.json](configs/warm_start_checks.json) 的后续对照也没有显示坐标预训练带来到达优势。

[supervision_audit.py](src/supervision_audit.py) 读取有限地图 checkpoint，分别测量完整距离、排序、近程与远程监督的梯度及方向关系。[supervision_use.json](configs/supervision_use.json) 从同一有限地图出发，只检验显式排序项是否妨碍距离拟合；全部距离标签保留。[dense_supervision.json](configs/dense_supervision.json) 再检验同一距离标签能否更有效地传递给各坐标：保留硬最大距离 Huber，另对每个超过真实距离的坐标差施加惩罚。这不指定坐标含义，也不改变 L∞ 精确可行解集合。两种诊断均以相同完整训练距离 Huber 选模，报告最后一步，测试不参与选择。

这几项小检查沿用 `structure_study` 的命令，将 `--contract` 换为对应合同，并用 `--pair-pretrained` 指向已完成完整关系组的 `best.pt`。初始化 checkpoint 的源码提交身份会与合同核对。它们只定位有限集合的优化问题；全空间到达仍由独立固定度量队列评估。当前数字与结论统一见[结果报告](results/report.md)。
