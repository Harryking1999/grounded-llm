# 主指标与最终轨迹图

训练进度和轨迹推进回答不同问题。三张学习曲线检查接口训练后的能力变化；两张最终 checkpoint 图直接比较参考历史与自身历史下的动作，并展示实际执行后何时到达、何时走入死局。清空／非清空只在[报告](../report.md)中保留原表格。

| 图 | 横轴与同图比较 | 范围 |
|---|---|---|
| [闭环学习曲线](rollout_learning_curves.png) | epoch；两任务 × 到达率／到达且最短路率 | 非零步 validation：寻路 500，积木 1,000 |
| [reference 学习曲线](reference_learning_curves.png) | epoch；两任务 × 动作可达率／最短路动作率 | 作答前仍可解的非终止参考状态 |
| [积木最终轨迹](blocks_final_trajectory_steps.png) | 决策步的 reference／rollout 对比；执行步的累计到达与首次死局 | epoch 5；累计图固定 1,000 个非零步任务 |
| [寻路最终轨迹](path_final_trajectory_steps.png) | 决策步的最短路动作率；执行步的累计到达 | epoch 3；累计图固定 500 个非零步任务 |
| [地图动作一致性](map_reading_learning_curves.png) | epoch；两任务的地图最优动作选择率 | 全部非终止参考状态，含已死局状态 |

## 绘图选择

版式参考 ICLR 2021 [DreamerV2 的学习曲线与消融图](https://arxiv.org/pdf/2010.02193)：同一指标使用对齐子图与共享图例。蓝色 `#0072B2`、橙色 `#E69F00` 和朱红色 `#D55E00` 取自 [Okabe–Ito 的 Color Universal Design 建议](https://jfly.uni-koeln.de/color/)，用点形和线型补充颜色区别。图内只保留标题、坐标、图例，分母和解释放在图外。

学习曲线中模型为蓝色圆点，地图贪心为橙色虚线，随机为灰色点线；最终逐步图中 reference 为蓝色圆点、rollout 为朱红色方点，地图贪心仍为橙色虚线。每步可解案例少于 30 时只画空心点，不与主曲线相连；这是低支持度提示，不是显著性检验。寻路几乎重合的动作可达率和恒为零的实际死局不再重复画图。

学习曲线只连接已有整数 epoch，不加入通路检查的两题 epoch 0；逐步图保留原轨迹步数，不做分箱和平滑，累计曲线采用阶梯线。纵轴从 0 起；闭环学习曲线的最短路率一行显示 0–50%，其余百分比图显示 0–100%（顶部留空）。当前只有一个训练运行，不画多次运行误差带。

## 数据与分母

- [原运行摘要](../long_trajectory_summary.json)：寻路 epoch 1–3、积木 epoch 1–4 的完整 validation 电池与整数计数。闭环曲线减去已保存的零步组分子和分母。
- [积木逐轮证据](../blocks_rollout_turn_analysis.json)：积木 epoch 3–5 完整主指标和地图一致性；重叠时使用完整复核结果。零步组均为 100/100 到达且最短，由[目标分层证据](../blocks_target_analysis.json)核对。
- [一步参照摘要](../learning_curve_baselines.json)：原 validation 状态上的随机与地图贪心参照，不运行 LLM。
- [最终轨迹证据](../final_trajectory_step_analysis.json)：两任务 final 的全部逐步计数，原示范与模型轨迹的累计到达、累计首次死局；仅读取已完成原案例和物理示范。
- [绘图计数](learning_curve_data.json)：上述证据合并后的主图分子、分母；同时保留 `rollout_all` 的含零步原读数。

**动作图的分母随步骤变化，累计图的分母固定。**动作图只纳入当步仍可解的非终止状态，无效输出计错；已死局状态排除。reference 与 rollout 第二步起访问不同物理状态和历史，其差值是模式差距，不是同状态的因果消融。晚步样本还受完成退出与存活筛选影响。

累计到达、首次死局以全部非零步任务作分母，不因任务结束而缩小。首次死局只统计实际执行的合法动作：积木 rollout 为 832/1,000，另 12 题未到达但未执行过破坏可达性的动作；寻路没有实际死局，109/500 未完成。地图贪心累计曲线来自原物理示范；不能把 reference 模型的错误当作已经执行，也不能将各轮正确率连乘制造 reference 成功率。寻路预算末轮的第 33 次决策可评分，但最多实际执行 32 个动作。

闭环贪心辅助线与主图匹配非零步范围：寻路到达 500/500、最短 115/500；积木到达 683/1,000、最短 311/1,000。一步随机参照先计算各状态的合格候选比例，再对状态取平均，地图最优并列项全部计入。贪心与随机均是参照而非严格上下界；尚无完整随机 rollout 数据，因此任务图不画随机成功率。

## 重绘与导出

从仓库根目录运行（绘图只需 matplotlib；本次使用 3.10.6，已保存摘要足够）：

```bash
python -m pip install matplotlib
python -m experiments.flamingo_map_reader.src.plot_learning_curves
```

如需从 Git 外原运行重算逐步摘要（使用项目环境，不运行模型推理）：

```bash
python -m experiments.flamingo_map_reader.src.collect_trajectory_steps \
  --run-root RUN \
  --out experiments/flamingo_map_reader/results/final_trajectory_step_analysis.json
```

重算一步参照的入口为 `collect_plot_baselines`，输出 `results/learning_curve_baselines.json`。每图同时保存 PNG（预览）、PDF（矢量排版）和 SVG（可编辑文字），文件同名；不再生成目标分层折线图。
