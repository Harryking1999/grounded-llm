# GCML 原始任务、设置与用例

核对日期：2026-09-04。这里记录参考实验的来源与核对结果；项目进度仍以 [项目状态](PROJECT_STATUS_AND_TODO.md) 为准。已完成 Luna 英文纯语言小批与直接兼容 API 的 Blocks 隔离复测；尚未复现 GCML 训练。

## 权威来源

- [正式论文：Neural sampling from cognitive maps enables goal-directed imagination and planning](https://www.nature.com/articles/s42256-026-01254-4)：Fig. 3 为 32 节点图任务，Fig. 4 为二维积木轮廓分解，Fig. 5 为扩展变体。
- [官方代码及数据](https://github.com/LH-cbicr/GCML/tree/ff76859b71a2bc2056b50f5e052475351c007f76)：本次检查版本为 `ff76859b71a2bc2056b50f5e052475351c007f76`，与查询到的 `GCML` tag 相同。
- [论文指定的 Zenodo 存档](https://zenodo.org/records/19370442)：链接到上述 `GCML` tag。代码采用 MIT 许可；下载副本中保留原始 `LICENSE`。
- [补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-026-01254-4/MediaObjects/42256_2026_1254_MOESM1_ESM.pdf)：Section D、Algorithms 1–2 描述图与轮廓任务推演。已读取网页提取文本；PDF 图像下载返回 HTML，未完成补充页的视觉核对。

以下 notebook cell 编号均从 0 开始。设置分为论文叙述、发布代码和本次数据检查，不将三者视为自动一致。

## 二维轮廓分解（积木）

### 任务与动作

在 10×10 二值网格中，`1` 表示轮廓占据的像素。每步从剩余轮廓移除一个指定形状、指定位置的积木，目标是清空轮廓。合法动作要求积木覆盖的像素当前全为 `1`；移除后置零。没有重力、三维抓取或物理抽出路径的约束。

官方 [积木 notebook](https://github.com/LH-cbicr/GCML/blob/ff76859b71a2bc2056b50f5e052475351c007f76/gcml_tiling.ipynb) 的 cell 1–2 和 [数据生成器](https://github.com/LH-cbicr/GCML/blob/ff76859b71a2bc2056b50f5e052475351c007f76/utils/dataset_tiling.py) 定义了八个带方向的形状：

| shape_id | 二值矩阵（`/` 分行） | 合法摆放位置数 |
| --- | --- | --- |
| 0 | `11/10` | 81 |
| 1 | `10/11` | 81 |
| 2 | `11/01` | 81 |
| 3 | `01/11` | 81 |
| 4 | `1/1` | 90 |
| 5 | `11` | 90 |
| 6 | `1/1/1` | 80 |
| 7 | `111` | 80 |

总计 664 个动作。没有 1×1 方块；四个拐角已经包含四种方向，不是每个拐角再旋转出四种独立类型。形状可以重复使用，代码不维护“每种只能用一次”的库存。

动作 ID 按形状、行、列依次枚举。坐标为从 0 开始的 `(row, column)`，锚点是积木外接矩形的左上角，行向下、列向右。动作是否合法只取决于对应掩码是否包含在当前轮廓内，不能限定为必须恢复数据生成时的那一种分解。

### 数据文件与切片

两份 [官方数据文件](https://github.com/LH-cbicr/GCML/tree/ff76859b71a2bc2056b50f5e052475351c007f76/dataset) 均已下载并读取。HDF5 group 为 `tiling`，字段为 `pre_obs`、`post_obs`、`action`、`element_idx`，原文件 dtype 均为 `float64`。

| 文件 | 原始轨迹数 | 每条轨迹的移除步数 | pre_obs / post_obs 形状 | notebook 使用方式 |
| --- | --- | --- | --- | --- |
| `tiling_10x10_5obj.h5` | 20,000 | 5 | `(20000, 5, 100)` | cell 3 取 `[0:18000]` 训练；按 loader 规则，剩余 `[18000:20000]` 可作同规模测试 |
| `tiling_order_10x10_8obj.h5` | 20,000 | 8 | `(20000, 8, 100)` | cell 7 取 `[18000:20000]`，共 2,000 条，用于从 5 块到 8 块的测试 |

`action` 与 `element_idx` 的形状分别为 `(20000, 5)` 或 `(20000, 8)`。单题输入是 `pre_obs[row, 0].reshape(10, 10)`；存档参考解是 `action[row]`，终点为 `post_obs[row, -1]` 的全零网格。参考解只供裁判核验，不能连同题目一起交给被测模型。

当前 notebook 的演示调用设置为 `traj_len=8, traj_num=10, test_single=True, test_dataset_idxs=[2]`：实际只演示 8 块测试切片中的第 3 题，即原文件行 18002。直接运行这一调用不能取得整份测试集的成功率。最大 8 步属于该调用的预算；原始参考分解有 8 步，不代表唯一解或最短解。

### 一个确切原始用例

文件 `tiling_order_10x10_8obj.h5`，原始行 18002（notebook 演示题）：

```text
0000000000
0000000000
1011000000
1111100000
1110111000
0011001000
0011011000
0001000000
0000000000
0000000000
```

存档参考动作 ID 为 `[524, 355, 20, 546, 444, 557, 620, 293]`，对应 `(shape_id, row, column)`：

```text
[(6,2,0), (4,3,1), (0,2,2), (6,4,2),
 (5,3,3), (6,5,3), (7,4,4), (3,5,5)]
```

本次已逐步核验：所有动作合法，每步结果与 HDF5 的 `post_obs` 完全一致，最终网格为空。

### 影响实验解释的原实现细节

- 论文 Methods 的 affordance 小节明确让积木任务逐步获取真实合法性信息。发布 notebook cell 6 在候选轨迹生成中调用 `env.gain_affordance_vector` 与 `env.check_affordance`，再令 `loc = next_state`；该循环没有用学到的 `V` 更新棋盘。因此这一发布实现不能直接作为“虚拟后继状态由学到的转移模型产生”的证据。这里需要区分规则程序在候选推演中的作用，与实际动作执行后环境返回真实观测的正常交互；后者符合本项目设想，不能据此否定内部想象或认知地图的价值。
- 正文 Methods 指出该任务使用 `Q=I`；发布实现以 `Q_encoder` 的固定卷积计算周边空白带来的偏好。这里的状态表示包含任务结构，不能据此认为原文已经学得通用状态编码器。
- 发布 notebook 默认 `load_trained_model=True`，尝试读取 `tmp_data/fig4_tiling.pth`，但该 checkpoint 不在本次官方源码包中。查看题目不需要该模型，也不应为提取用例启动训练。
- 论文称测试时排除与训练相同、仅平移位置的轮廓。本次将初始轮廓裁切到非零像素的最小外接矩形后比较：5 块文件的最后 2,000 行中，有 18 行与前 18,000 行相同；例如测试行 18103 对应训练行 10647。8 块测试切片按相同准则与 5 块训练集比较，发现 0 行重合。这是对发布数据的检查，不等于复核了论文作图使用的全部中间文件。

## 32 节点抽象图

### 原图与目标

[图任务 notebook](https://github.com/LH-cbicr/GCML/blob/ff76859b71a2bc2056b50f5e052475351c007f76/gcml_abstract_graph.ipynb) 的 cell 5 保存了带有“the abstract graph used in the paper”注释的完整邻接矩阵。代码先调用随机图生成器，随后用该固定矩阵覆盖，因此默认运行使用原图。

本次读取并确认：32 个节点、48 条无向边、96 个有向动作；图连通，每个节点的度为 2–5。观测是节点 one-hot，动作是沿一条边的特定方向移动；同一无向边的两个方向是不同动作。动作 ID 先扫描上三角，再依次编号正向与反向。

cell 15 的具体示例为 `start_idx=26, goal_idx=1`，节点编号从 0 开始。本次用独立 BFS 算出一条最短路径：

```text
26 -> 27 -> 31 -> 20 -> 7 -> 6 -> 5 -> 1
```

长度为 **7 条边**。这一参考答案是本次从官方邻接矩阵计算的，不是 GCML 训练后的输出；节点个数不能当作路径长度。

### 正文与发布 notebook 的设置区别

正文依据为 Fig. 3 及 Methods 中 “Further details for the application of GCML for goal-directed imagination for generic problem-solving”；代码依据为 cells 4、5、8、10、14、15。

| 项目 | 正文 | 发布 notebook |
| --- | --- | --- |
| 探索轨迹 | 200 条，长度 32 | 配置生成 400 条，长度参数 128；随机游走函数实际输出 127 个转换 |
| 状态维度 | 该任务量级约 1,000 | `state_dim=1000` |
| 训练轮数 | 此处未明确给定 | `epochs=5000`；本次未运行 |
| 图路径样本数 | 每组 40 条 | 演示 `theta_cycle_num=100` |
| 推演上限 | 补充算法使用长度预算 | 演示 `traj_len_per_theta_cycle=200` |
| 噪声 | Fig. 3 对比 0.15 / 0.25 | 演示 `sigma=0.1`，按当前 utility 最大值缩放 |
| 节点奖励 | `Uniform(-5,5)` | 初始化为 `Uniform(0,10)`，标记到达时再减去 `5 × 动作数` |
| 合法动作信息 | 起点真实合法性，后续估计 | 含噪分支起点用环境表，后续用 `G @ s_pre`，并记录已选动作 |

发布代码同时记录内部状态、动作列表和映射得到的节点列表，到达判据与记录动作的时间位置也需在正式复现前核对。提取图用例及计算合法参考路径不依赖这一训练／采样循环。本次没有修订原算法或声称复现 Fig. 3 数值结果。

正文 Fig. 6 的 `k=5`、多种图规模对比是另一组设置，不能替代 Fig. 3 的 32 节点演示合同。若后续开展奖励筛选测试，还需明确路径长度预算、是否允许重复节点、奖励计数规则及筛选预算。

## 本地提取结果与后续使用

完整官方副本在忽略目录 `literature_review/gcml_archive/GCML-ff76859b71a2bc2056b50f5e052475351c007f76/`。本次只提取了小批示例，没有重新生成同类题或调用模型：

- [32 节点原图、动作字典和参考路径](../data/raw/gcml_examples/graph32_official_demo.json)。
- [5 块原始样例](../data/raw/gcml_examples/tiling_10x10_5obj_examples.json)：行 18000–18002。
- [8 块原始样例](../data/raw/gcml_examples/tiling_order_10x10_8obj_examples.json)：行 18000–18002，包含 notebook 演示题。
- [积木形状、坐标规则与 664 个动作映射](../data/raw/gcml_examples/tiling_action_catalog.json)。
- [读取与核验摘要](../data/raw/gcml_examples/inspection_summary.json)：包含数据形状、重复轮廓行号和验证范围。

上述 JSON 与下载数据均为本地忽略产物，不随 Git 文档提交。检查脚本位于 `literature_review/gcml_inspection/extract_examples.py`；它只解析 notebook 的矩阵字面量、读取 HDF5、核验 6 条积木参考轨迹并运行一次图 BFS，不执行 notebook。

建议下一步先将 8 块测试切片与这张原图转为英文输入，明确每轮推理、实际执行与反馈的协议。模型获得初始轮廓／图信息、规则和目标，在推理时先想象后选择动作；环境执行实际动作后返回约定的新观测，模型据此校准状态并继续推理。基础模型与状态增强模型应获得同样的实际反馈，解序列留给裁判。

推理阶段是否能调用精确转移、合法动作查询或搜索代码，应独立于真实执行后的反馈来记录。“无工具推理”仍可包含真实动作后的正常环境反馈；一次性输出完整解且中途不获反馈，只是可选诊断形式，不是本项目的唯一任务定义。5 块题可作较近的基线，但在主张 GCML 训练外泛化前应处理已发现的轮廓重合。
