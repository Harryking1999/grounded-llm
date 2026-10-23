# 三个新任务的 Qmap 实验

这里训练和评测推箱子、移车出库、棋子交换的地图。三个任务各自训练，每题只给一个完整目标；先检查地图能否独立贪心完成，再考虑接入LLM。

**三个任务目前均未达标。** 当前成绩、已查明的问题和未完成事项统一见[结果报告](results/report.md)。

## 任务与配置

| 任务 | 规则与目标 | 数据 | 基线配置 |
|---|---|---|---|
| [推箱子](../../tasks/sokoban/README.md) | 10×10四箱，箱子和玩家都到指定位置 | [说明](../../tasks/sokoban/data/README.md) | [配置](configs/sokoban_pairs.json) |
| [移车出库](../../tasks/rush_hour/README.md) | 无墙6×6，所有车辆都到指定位置 | [说明](../../tasks/rush_hour/data/README.md) | [配置](configs/rush_hour_pairs.json) |
| [棋子交换](../../tasks/token_swapping/README.md) | 固定10顶点树，匹配指定完整排列 | [说明](../../tasks/token_swapping/data/README.md) | [配置](configs/token_swapping_pairs.json) |

推箱子和移车出库检查未见布局；棋子交换检查固定树上的未见状态与关系。各必需组的完整贪心完成率至少80%，目标90%以上，具体见[验收配置](configs/exact_pair_acceptance.json)。

评测保留全部合法动作，不搜索、不重试、不自动排除死路。环境给出真实后继，地图只比较它们到目标的预测距离。完成率之外，还报告距离误差、最优率、循环和死锁。

## 已开展的对照

| 任务 | 检查的问题 | 配置入口 |
|---|---|---|
| 推箱子 | 输出形式、近距与死锁样本、棋盘编码、不可达样本的训练数值 | [Linear](configs/sokoban_linear_output_pairs.json)、[采样](configs/sokoban_balanced_relations_pairs.json)、[卷积](configs/sokoban_spatial_pairs.json)、[训练数值](configs/sokoban_proxy128_pairs.json) |
| 移车出库 | 棋盘编码、车辆字母命名的影响 | [卷积](configs/rush_hour_spatial_pairs.json)、[车辆编码](configs/rush_hour_lane_order_pairs.json) |
| 棋子交换 | 近距离样本、动作排序、距离计算方式 | [补近](configs/token_swapping_local_goals_pairs.json)、[排序权重4](configs/token_swapping_local_rank4_pairs.json)、[排序权重8](configs/token_swapping_local_rank8_pairs.json)、[更近目标](configs/token_swapping_local_walk8_rank4_pairs.json)、[分组L1](configs/token_swapping_grouped_l1_pairs.json) |

结果以报告为准，配置存在不代表已经完成；Softsign仅保留候选配置，未运行。所有参数只在配置中维护。

## 推箱子覆盖量对照

先比较[64张训练棋盘](configs/sokoban_coverage64_pairs.json)与[1,000张训练棋盘](configs/sokoban_coverage1000_pairs.json)。两组使用相同模型、总训练步数和100张验证／100张测试棋盘，按棋盘均衡采样，并保留近距离与死锁边界监督。新增训练棋盘包含原来的64张，具体名单见[选题配置](configs/sokoban_coverage_layouts.json)。

每张棋盘限制训练起点数，精确距离仍来自完整物理图。这个对照回答扩大布局覆盖是否有用；与旧基线相比，采样方案也有变化。

扩大棋盘数量后，原评测按精确步数从短到长分配，会使每张棋盘只测到很短的题。覆盖量比较因此使用同一批参考题，按固定距离区间重评两个冻结模型；原始成绩保留。训练与验证选权重不变，最终比较见报告。

## 运行入口

以棋子交换为例，先准备数据，再训练：

```bash
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --prepare-only
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/token_swapping_pairs.json --out runs/new_tasks_pairs/token_swapping --source-commit COMMIT --device cuda:0 --resume
```

另两任务替换配置和输出目录。正式运行使用已提交源码，先按[机器手册](../../DEVELOPMENT_MACHINES.md)核对机器、进程和资源。恢复训练需要完整保存的状态，不能只加载推理权重。

已有模型可单独评测；更多选项见对应模块的`--help`：

```bash
python -m experiments.new_tasks_qmap.src.relation_evaluate --checkpoint runs/new_tasks_pairs/token_swapping/best.pt --run runs/new_tasks_pairs/token_swapping --out runs/new_tasks_pairs/token_validation.json --cohorts validation unseen_endpoint_validation unseen_both_validation
```

## 文件放在哪里

- `configs/`：正式实验参数与验收要求。
- `results/`：报告和紧凑结果；失败对照同样保留。
- `runs/new_tasks_qmap/`：原始数据、日志、训练恢复文件和运行记录，Git忽略。
- `tasks/<task>/Qmap/selection.json`：通过验收后采用的权重；三个新任务目前均未采用。

实现入口为[src/run.py](src/run.py)，规则和数据适配在[任务实现](../../grounded_llm/tasks/)，公共方法见[Qmap说明](../../tasks/QMAP_METHOD.md)。

## 历史试验

早期手电过桥、小棋盘推箱子、24点的条件与当前三任务不同，保留[首轮结果](results/summary.json)和[诊断](results/diagnosis.json)，配置分别为[过桥](configs/bridge.json)、[小棋盘推箱子](configs/sokoban.json)、[24点](configs/game24.json)。不纳入当前地图验收。
