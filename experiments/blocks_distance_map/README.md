# Blocks distance-supervised Q-map

本目录供执行与复核实验的 agent 使用。面向人的说明见 [BLOCKS_QMAP_PLAN.md](../../docs/BLOCKS_QMAP_PLAN.md)。当前进度只维护在 [PROJECT_STATUS_AND_TODO.md](../../docs/PROJECT_STATUS_AND_TODO.md)；本文件定义执行协议，不创建另一份当前状态页。

## 1000 棋盘共享 Q：多目标与跨棋盘检验

新一轮使用 [multiboard_1000_multigoal.json](configs/multiboard_1000_multigoal.json) 作为唯一正式参数合同。`src/multiboard_data.py` 从原论文八形状数据集按固定 seed 选出互不重复的训练与 OOD 初始棋盘；参考动作逐步重放，随机合法移除形成分支状态树。地标来自不同剩余格子数量的非空中间状态，显式包括含孤立格、无法清空但可从更早状态到达的目标。`DistanceOracle.distance(当前, 目标)` 仍用精确差集覆盖给出有向有限距离或已证明不可达；无法清空只描述到空棋盘的关系，不作为状态的固定坏标签。

同一共享棋盘编码器 `Q(棋盘)` 输出 128 维向量；目标进入固定有向距离 `D(Q(当前), Q(目标))`，不进入编码器。训练保留不可达距离惩罚，并联合优化距离 Huber 与按相同起点或目标组成的 listwise 排序。部分训练棋盘加入“同一父状态、两个合法后继、两个非空目标，换目标应换动作”的四关系监督；另一些训练棋盘保留同类对照仅供测试。候选后继由规则环境提供，不用 Q 搜索或选择训练标签。

评测分开记录：训练棋盘内见过状态但未见过的状态对、同棋盘从未进入训练关系的状态、训练中从未作为目标的地标、其中含孤立格的目标、以及初始棋盘完全未进入 1000 张训练集的 OOD 棋盘。留出地标同时取含孤立格与不含孤立格两类，前者单独统计。验证集只包含同训练棋盘中未见的状态，选 checkpoint；其余测试不参与选模型。换目标对照还分别报告两个目标的一步排序、双目标都正确的反转率，以及从同一父状态分别到两个非空目标的逐步贪心到达率。另从可达起点逐步走向含孤立格的未见目标。所有测试关系使用精确标签评测，推理分数只来自冻结 Q。状态掩码可能在不同初始棋盘的分支中重合，OOD 关系要求至少一个端点未进入训练状态池；摘要记录实际分母。

运行需要 `numpy`、`torch` 和 `h5py`。先将原论文的 `tiling_order_10x10_8obj.h5` 放到配置指定的忽略路径 `data/raw/`，从仓库根目录执行：

```bash
python -m unittest discover -s experiments/blocks_distance_map/tests -v
python -m experiments.blocks_distance_map.src.multiboard_run \
  --config experiments/blocks_distance_map/configs/multiboard_1000_multigoal.json \
  --out runs/blocks_distance_map/RUN_ID --source-commit COMMIT --device cuda:0
```

正式运行从已提交的 commit 复制源码到独立远端目录，脱离 SSH 会话执行。`data/data.npz` 保存各关系组、棋盘行号与换目标对照；含孤立格测试是未见目标测试的显式子集。`progress.jsonl` 保存验证轨迹；`best.pt` 是按验证目标选择的权重；`checkpoints/step_*.pt` 是含优化器的周期检查点；`summary.json` 是最终测试。生成物均在忽略的 `runs/`。下文记录此前单棋盘试点的合同与结果，不作为这次共享 Q 的测试数据。

## 既有单棋盘试点：假设与范围

对同一固定积木棋盘中的状态，直接监督精确有向最短步数与不可达关系，能否形成支持距离排序与好坏候选判断的紧凑 Q-map？先检验关系拟合与留出关系补全，不宣称未见状态或跨棋盘泛化。

用户已授权在指定开发机训练，并明确要求原论文较简单的形状集合。环境复用 `experiments.gcml_counterexamples.src.blocks` 的官方八形状规则，共 664 个带位置动作；不得换回 `sol_dag_blocks` 的十形状变体。棋盘、原始数据来源、参考动作、采样预算、优化设置的唯一机器可读合同是 [configs/pilot.json](configs/pilot.json)。参考路径须逐步重放，但不当最短路标签。

本轮只训练 Q。不训练 V、LLM、分类头、策略网络或目标条件编码器，不运行对外 API。精确 solver 只负责离线标签及 oracle 评测。

## 实现接口

| 文件 | 职责 |
| --- | --- |
| `src/oracle.py` | 复用官方动作目录；差集精确覆盖 DP；返回有限距离或 -1（已证明不可达）；预算耗尽抛出 unknown 异常 |
| `src/data.py` | 构造解与随机合法 rollout；精确 pair 标签；状态对分组划分；固定起点／终点的排序对；留出目标候选集 |
| `src/model.py` | 目标无关的 Q 查表或完整棋盘 MLP；固定有向距离及欧氏对照；采样分层 |
| `src/train.py` | 只优化 Q；距离 Huber 加排序 hinge；验证集早停；保留最优 checkpoint |
| `src/evaluate.py` | 有限距离、排序、不可达与同父状态候选评测；面积和 oracle 基线 |
| `src/diagnose.py` | 统计训练约束覆盖与留出目标关系的距离误差；同时支持查表与棋盘编码器 |
| `src/rollout.py` | 用真实合法后继与 Q 分数逐步贪心；查表版缺候选即停，编码器版对新棋盘直接编码 |
| `src/train_encoder.py` | 在相同精确状态对监督下训练棋盘编码器；不使用目标条件输入或 V |
| `src/run.py` | 一份共享数据，按合同顺序跑全部 metric 与 seed，并自动汇总 |
| `tests/test_distance_map.py` | 标签器与穷举 BFS 比较、坏合法分支、预算语义、方向表达、划分与平局评分 |

`d(s,t)=-1` 仅表示严格不可达。若 `t` 不是 `s` 的子集可直接判负；否则计算 `cover(s XOR t)`。选择一个格子枚举覆盖它的合法 tile，缓存剩余 mask 的最小分解数。只有完整失败才能缓存 -1；预算异常必须向上传播，不能转换成标签。形状可重复、无库存、无重力且移除可交换，是差集公式成立的条件。

## 数据与泄漏边界

1. 包含原参考解、随机合法路径上的全部状态与边；不移除通往死局的边。
2. 从已采到的可解状态随机选决策父状态，将它们所有合法后继加入状态池。使用可解父状态是明确的条件式动作评测，不代表随机所有状态的总体分布。
3. 每个非空状态与空棋盘的两个方向均有标签。加入全部采样边、决策边及反向对、轨迹中的多步对、困难子集负样本与随机状态对；实际可获得的分层数量以数据摘要为准，不伪造配额。`target_pairs` 是软目标，必须保留的覆盖／决策关系可使总数超过它，摘要记录实际数量。
4. 将初始棋盘与每个其他状态的双向关系保留在 train，确保每个 Q 行都有训练约束；此集合包含初始棋盘到空棋盘的标签。其他关系按无序状态对分组划分。决策后继到空棋盘及反向关系强制进入 test，不能与初始锚点关系冲突。任何排序比较只能引用同一 split 的 pair 行。
5. 后继到目标的留出标签不能用作训练损失、排序损失、checkpoint 选择。后继 Q 可从其他训练关系中学习；这一设置是 transductive relation completion。
6. 验证训练约束覆盖所有 Q 行；若不足，失败并检查数据，不能把保留的目标标签挪回训练来补洞。
7. 求解器完整结束才写出可用于训练的数据文件。标签器超时是未完成，不是训练完成或不可达。

首轮已发现：随机状态对几乎不会产生“源状态包含非空目标、但差集不可分解”的困难负样本；决定性后继状态多数也没有训练过的向外一步关系。`landmark_probe` 因此增加所有位于初始棋盘内的单块棋盘作为非空地标，并为每个状态采样若干个它包含的地标，精确标注到地标的距离或不可达。所选地标关系进入训练，直接到空棋盘的决策后继标签继续留出。这是更充分的监督条件，不能同首轮的 100,000 对设置视作仅增加训练步数。

有向 `max` 距离只有取到最大坐标的分量接收单个样本的梯度。若 landmark 条件下仍难优化，`dense_direction_probe` 在完全相同的数据上改用 `sum_i relu(q_s[i]-q_t[i]) / sqrt(dimension)`；它仍有方向且满足三角关系，但梯度可以经过更多坐标。此对照检验距离公式的优化差异，不是更改状态、标签或训练目标。

`sparse_direction_ablation` 在首轮无非空地标的数据上单独训练有向求和距离，使用相同结构和训练预算，用于区分“距离公式改善”与“非空地标监督增加”的作用。首轮与地标数据的状态池并非完全相同，但决策父状态由相同采样前缀选出；它是机制诊断，不能替代严格固定全部状态对的因果对照。

`sparse_encoder_ablation` 再把完整棋盘 MLP 用于同一份稀疏数据，与 `encoder_probe` 的地标数据版本比较，以检查共享棋盘编码本身是否足以替代非空目标监督。架构、距离形式和训练预算保持一致。

`second_board_replication` 使用另一张原论文八块棋盘、相同规则和同样的非空地标监督，在该棋盘单独训练 Q。它检验实验流程能否在另一题上重复；并非一个 Q 编码器直接推广到新棋盘。由于首张棋盘的测试结果已用于选择有向距离形式，第二张棋盘的复测更适合检验该选择，不能把两张棋盘说成大样本泛化。

`encoder_probe` 保留首张棋盘的原始状态对、标签和划分，只把每个 state id 的独立参数换成读取 100 个占用位的共享 MLP。它可以为数据池以外的残余棋盘计算 Q；逐步贪心是实际检验。训练期的 pair 验证和测试仍共享许多已见状态，不能称为未见棋盘测试。先前探索已反复查看测试结果来决定监督和距离形式，本轮测试也应标作探索性诊断。

`second_board_encoder` 在第二张棋盘上用同一结构、同一训练预算及其独立数据训练新的编码器，检验首张棋盘的结果能否在另一个单题条件下重现。另做第一张棋盘编码器对第二张棋盘的零样本逐步诊断；这与第二张棋盘的重新训练是两种不同条件。跨棋盘评测时必须把第一张棋盘的数据文件作为 `rollout --training-data`，才能正确统计离开训练状态池的棋盘。

保存 `data.npz`：十进制字符串 state masks、`pairs=[source_id,target_id,distance,kind]`、split、各 split 的 outgoing/incoming 近远 pair-row 索引、合法转移、目标标签和独立候选决策表。kind=0 有限、1 非子集负样本、2 子集但无法分解。目标标签和候选表仅用于标注/评测，不能当模型输入。

## 模型与目标

查表 Q 为 `Embedding(state_id)`；编码器 Q 为 `MLP(100 个棋盘占用位)`。查表对照使用同维度与匹配的随机初始化，距离形式分别为 `max_i relu(q_s[i]-q_t[i])` 和 L2；后续还比较更易优化的有向求和距离。编码器沿用有向求和距离，不输入目标。所有 Q 行（含空棋盘）都训练，不固定 Q(goal)=0。

`H=floor(initial_occupied_cells/min_tile_cells)` 是有限合法路径的统一上界；不可达目标取 `C=H+1`。坐标每步投影到 `[0,C]`，有向距离因而有界；欧氏对照也用相同坐标边界，但其距离本身可能超过 C。

距离损失是 `smooth_l1(D, d_or_C)`；排序损失为 `relu(margin+D_near-D_far)`。距离项的有限／不可达家族各占一半权重，在家族内对实际存在的距离层／负样本类型等权。排序项对 outgoing／incoming 两类等权，近远必须有严格不同标签。总目标及具体 margin、权重见配置。

Adam 只接收 Q 的参数。验证集使用与训练同分层的距离损失加排序损失选择 checkpoint。测试指标只作报告，不能反复根据它挑 epoch 或修改超参数后仍声称测试未使用。初始 Q 的测试分数作为固定基线保存，不影响选择。

## 指标及解释

- 有限距离 MAE、四舍五入距离准确率、汇总 Spearman；汇总相关不是逐起点排序的替代。
- 固定起点与固定终点的排序准确率、平局率、hinge loss 分开报告。
- 不可达 AUROC，以及固定 `C-0.5` 阈值下的不可达召回、有限状态误报率、困难负样本召回。阈值指标需距离校准，不能用 AUROC 代替。
- 决策按 `D(Q(actual_successor), Q(goal))` 评分；只有合法候选，后继来自真实环境。报告可解动作率、最优动作率、进入死局率，以及全部／同面积／深层死局的好坏配对排序。
- 最优动作指候选后继精确目标距离最小的任意动作；平局按均匀随机选择的期望值计分。配对排序要求严格排对，平局不算成功。
- 面积基线与精确距离 oracle 使用完全相同的候选。原十形状旧 Q/V 结果只作历史背景，不能称为本合同的匹配对照。
- 三个 seed 报告逐项值与均值/范围。单棋盘和大量相关状态对不是独立棋盘重复；不据此给跨棋盘显著性结论。
- 逐步贪心诊断从初始棋盘和决策父状态开始，每步使用环境真值枚举合法后继，仅用 Q 给后继评分；缺少任何候选后继 Q 时记录 `candidate_q_missing`，不把该候选删掉继续规划。它仍不是学得 V 的开放环规划。成功率同时给出完整分母与覆盖失败数。

是否推进到 V，参考面向人文档的暂定工程标准；若不达标，分别定位训练拟合、留出关系或候选排序。不得把验证未通过的 Q-map 自动接入 LLM。

## Q 坐标与转移图可视化

面向人的[可视化图例](results/qmap_geometry.md)区分冻结 Q 的二维投影、真实合法转移和按步骤排列的棋盘示意图。`src/export_geometry.py` 从报告中的一个既有案例出发，枚举全部后续状态与动作边，使用 checkpoint 中的棋盘编码器计算 Q，不更新参数。完整状态数超过预算时失败，不静默导出截断图。原始 Q 向量写入忽略的运行目录。

```bash
python -m experiments.blocks_distance_map.src.export_geometry \
  --checkpoint CHECKPOINT --training-data TRAINING_DATA \
  --case-summary experiments/blocks_distance_map/results/summary.json \
  --out runs/blocks_distance_map/GEOMETRY_RUN
python experiments/blocks_distance_map/results/plot_geometry.py \
  --export-dir runs/blocks_distance_map/GEOMETRY_RUN
```

绘图使用 Python、numpy、matplotlib；计算投影还需要 scipy 与 scikit-learn。PCA 输入原始 Q；t-SNE 输入两方向 Q 距离的均值，等于当前 `directed_sum` 的缩放 L1 距离。真实距离、可解性标签与边不用于拟合二维坐标。固定随机种子和参数保存于 `results/qmap_geometry.json`，该文件同时保留小图的全部节点、边、路径和二维坐标，不保存原始 128 维 Q。不带 `--export-dir` 运行绘图脚本，可直接从已保存的图数据重生成 PNG/SVG。

当前图例覆盖案例 A 的 336 个状态，而非 A 初始棋盘的完整图。绿线重放原评测的同起点、同种子贪心路径；红线强制第一个坏动作后继续贪心，必须明确它不是当前模型实际失败。放大图保留原 t-SNE 坐标，仅隐藏死局节点。邻居保留数以对称化 Q 距离为参照，不能解释成原始有向规划距离的保真度。

## 执行

Python 需 numpy 与 PyTorch；不需要 transformers。复用开发机现有环境，不为版本号重新安装整套依赖，实际版本写入运行摘要。

从仓库根目录执行最小正确性检查：

```bash
python -m unittest discover -s experiments/blocks_distance_map/tests -v
```

完整运行入口（路径与 commit 由本次运行明确指定）：

```bash
python -m experiments.blocks_distance_map.src.run \
  --config experiments/blocks_distance_map/configs/pilot.json \
  --source-commit COMMIT \
  --device cuda:0 \
  --out runs/blocks_distance_map/RUN_ID
```

需先提交正式源码和合同，再通过 `git archive` 将该 commit 的代码复制到独立远程目录。不得用预览代码跑正式训练并虚填 commit，不覆盖旧运行。可先对预览代码运行单元测试、标注和极短训练 smoke；它们不作为正式科学结果。

通过 nohup 脱离 SSH 运行并保存 PID、stdout 日志。队列自动按配置跑完并写总 `summary.json`。一次启动检查后，按实测速度决定下次检查时间，避免反复查询。机器地址、实际解释器、PID 等写入忽略的运行记录；与结果身份有关的 commit 和产物路径在结果报告中给出。

产物结构：`config.json`、`source_commit.txt`、共享 `data/`、各条件 `progress.jsonl` / `q.pt` / `summary.json`，以及队列总 `summary.json`。全都写入 Git 忽略的 runs；仅提交紧凑结果摘要与报告。当前任务不包含 git push 或 PR。
