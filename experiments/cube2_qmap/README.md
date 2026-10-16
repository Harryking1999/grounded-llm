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

## 完整路线与参数化对照

[route_complete.json](configs/route_complete.json) 与 [route_complete.py](src/route_complete.py) 实现下一项小型诊断：从八步起点和路径中间目标构造多目标关系，递归加入每个最优后继，直到目标；每个父节点保留全部六个动作候选。所有最优分支都闭合在受监督决策内，而非只监督一条选定路径。距离仍是完整状态集合的真实距离，网络自行组织坐标。

自由坐标组以共享编码器的随机初始输出初始化，并保留同一 sigmoid 输出范围；初始 Q 在合同浮点容差内匹配，实际最大差写入结果。两组使用同样的距离与排序标签、Adam、步数及学习率计划。参数化不同意味着相同学习率不保证相同的 Q 步长，因此记录实际 Adam 更新后的两项损失、Q 移动、距离关系与动作决策退化。验收同时检查距离平均／分位误差、所有受监督动作和完整起点路线；最大误差作为诊断，不再要求所有距离都低于同一个极小阈值。

```bash
python -u -m experiments.cube2_qmap.src.route_complete --contract experiments/cube2_qmap/configs/route_complete.json --prepared-run /PATH/TO/cube2_quarter_turn_8eed676 --out /NEW/RUN/seed_20261016 --source-commit COMMIT --seed 20261016 --device cuda:0
```

自由坐标仅用于有限集合诊断，不具备未见状态编码能力。覆盖路线评测中，两组均在首次进入不受监督的状态—目标决策时计为失败；从不屏蔽动作或用 oracle 纠正选择。共享编码器另行评测留出共同左乘状态集合，复用相同关系类型，不参与选模。

## 重叠多目标集合、保持与推广

[overlapping_routes.json](configs/overlapping_routes.json) 与 [overlapping_routes.py](src/overlapping_routes.py) 在 4090 上复现旧小地图，再从同一份权重分支比较：重叠集合交错联合训练、同批重叠集合顺序训练、无重叠集合交错联合训练。各集合共享完整路线模板，但重叠条件对齐真实父状态及其动作邻域，使新集合与某个旧集合共享物理输入状态；不只是复用编号。

三条件的每集合更新次数和组内全部距离／动作标签相同，联合条件在每一轮中各访问每集合一次。新阶段统一使用全新 Adam 与常数学习率。主结果取固定预算的末步；辅助最佳点只按训练集合平均目标选择。顺序条件中尚未训练的集合会被监测，但其标签不进入当前更新。重叠和无重叠条件的不同物理输入数是干预本身的一部分，不能宣称所有信息量均匹配。

评测分别报告旧集合保持、每个新集合拟合、已见状态之间未训练的新关系，以及共同的留出物理状态队列。新关系的父节点／候选到目标标签均不在任一训练集合的距离项或排序项中，所有输入却都见过；距离分层采用两个物理队列都具备的层及匹配计数。留出队列排除两种训练条件的全部输入，且不参与选模。记录实际新集合更新对旧集合损失、距离误差和动作选择的影响。

```bash
python -u -m experiments.cube2_qmap.src.overlapping_routes --contract experiments/cube2_qmap/configs/overlapping_routes.json --inputs /PATH/TO/cube2_overlap_inputs --out /NEW/RUN/seed_20261016 --source-commit COMMIT --seed 20261016 --device cuda:1
```

输入目录包含原精确 oracle、保护状态、已完成路线检查的 `base_data.npz`；权重按旧检查合同在目标机器重新训练，而非假定其他机器共享磁盘。集合个体的构造性可行性不证明超过 128 个状态的联合约束存在精确解。本轮保留原编码器、L∞ 和两项关系损失，不提供坐标答案。机器、产物及结果见[结果报告](results/report.md)。

[overlap_policy_eval.json](configs/overlap_policy_eval.json) 和 [overlap_policy_eval.py](src/overlap_policy_eval.py) 在训练结束后另行评测真正的完整贪心到达：允许中途进入任何合法状态，使用固定预算、全部六个动作，不用搜索或禁逆动作。分别保留旧图、全部训练图、跨目标新关系和留出物理状态的结果；与“首次离开受监督决策即失败”的覆盖路线指标区分。主比较仍取末步，不按新增评测重新选模。

## 全状态联合训练与关系留出

[full_state_joint.json](configs/full_state_joint.json) 与 [full_state_joint.py](src/full_state_joint.py) 实施全物理状态范围的联合训练。从已完成的重叠联合末步权重继续，重置 Adam；每遍无放回遍历全部父状态，并监督每个父状态的六条动作边。每个动作星形邻域同时面对一个按距离分层产生的目标和另一个邻域的目标；所有父／候选到目标的距离进入绝对距离损失，同批完整六候选进入排序损失。跨邻域目标持续变化，关系不局限于原八个集合。

固定重放轮流使用原八个集合的完整组内关系，以及一批新的全局父／候选目标关系。分别记录原图拟合、无中途限制的到达，以及新固定关系拟合。保留原共享编码器、128 维 L∞ 和两类损失；不提供坐标答案。

这是**新划分协议**：全部物理状态都允许训练，历史保护端点不再排除。验证、测试及历史队列的父节点与六个候选到目标的无序距离对，全部屏蔽于在线距离标签和排序标签，固定重放也须无交集。新任务从距离四步及以上抽取，避免留出动作边与全边训练冲突。遍历结束后再将主结果解释为“状态见过、初始关系未直接训练”；完整轨迹可以进入其他已训练关系，不能称为整条轨迹全部关系都未见。旧版未见状态结果只作历史记录，不混用协议。

```bash
python -u -m experiments.cube2_qmap.src.full_state_joint --contract experiments/cube2_qmap/configs/full_state_joint.json --inputs /PATH/TO/cube2_overlap_inputs --prior-study /PATH/TO/cube2_overlap_2f53baa/seed_20261016 --out /NEW/RUN/seed_20261016 --source-commit COMMIT --seed 20261016 --device cuda:0
```

主结果固定为完整预算后的末步，逐距离报告误差、单步最优和真实贪心到达／循环；十四步层的所有动作均可最优，因此整体单步率必须结合分层与到达率解释。记录精确的父状态、动作边及标签曝光数；曝光完整不等于全部约束拟合成功。三个暖启动模型使用同一采样流和关系队列，无新旧采样器匹配消融，故本轮是方案有效性检查，不能单独归因于某一新增采样组件。短 `--smoke-steps` 仅检查流程和吞吐，不满足全状态验收，也不覆盖正式产物。
