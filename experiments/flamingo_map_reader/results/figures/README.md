# 主指标学习曲线

四张图用于回答：训练读取接口后，LLM 是否更能读取地图、选择有效动作，并将这一能力用于完整任务。图内仅保留标题、坐标和共享图例；详细分母与解释放在[报告](../report.md)。

| 图 | 同图比较 | 范围 |
|---|---|---|
| [完整任务](rollout_learning_curves.png) | 两任务 × 到达率／到达且最短路率 | 全部 validation，含初始即目标 |
| [reference 动作](reference_learning_curves.png) | 两任务 × 动作可达率／最短路动作率 | 作答前仍可解的非终止参考状态 |
| [积木目标](blocks_goal_learning_curves.png) | 清空／非清空 × 两项闭环主指标 | epoch 3–5，排除初始即目标 |
| [地图动作一致性](map_reading_learning_curves.png) | 两任务的地图最优动作选择率 | 全部非终止参考状态，含已死局状态 |

## 绘图选择

版式参考 ICLR 2021 [DreamerV2 的学习曲线与消融图](https://arxiv.org/pdf/2010.02193)：将任务和指标放在对齐的子图中，使用共享图例，避免在一个坐标轴中堆叠不同指标。蓝色 `#0072B2`、橙色 `#E69F00` 和朱红色 `#D55E00` 取自 [Okabe–Ito 的 Color Universal Design 建议](https://jfly.uni-koeln.de/color/)，同时用圆点／方点及实线／虚线／点线传达区别。

模型为蓝色圆点实线；贪心为橙色虚线；随机为灰色点线。目标分层图将颜色用于清空和非清空，线型用于模型和对应目标组的贪心。横轴只标已有整数 epoch；不加入通路检查的两题 epoch 0，不平滑、不外推。纵轴从 0 起，同图同一行的两任务采用相同范围；闭环最短路率显示 0–50%，目标分层图分别显示 0–80% 和 0–40%。当前只有一个训练运行，故不画多次运行的误差带。

## 数据与参照线

- [原运行摘要](../long_trajectory_summary.json)：寻路 epoch 1–3、积木 epoch 1–4 的完整 validation 电池与整数计数。
- [积木逐轮证据](../blocks_rollout_turn_analysis.json)：积木 epoch 3–5 完整主指标和地图动作一致性。与旧摘要重叠时使用此完整复核结果。
- [目标分层证据](../blocks_target_analysis.json)：积木 epoch 3–5 清空与非清空的完整任务计数；未补造 epoch 1–2 的目标分层点。
- [参照摘要](../learning_curve_baselines.json)：从原 validation 示范与已保存的 final reference 状态离线计算；不生成 LLM 回答。
- [绘图计数](learning_curve_data.json)：以上证据合并后的全部分子、分母与参照值，供核对和重绘。

reference 随机参照是每个状态中合格候选数／全部合法候选数，再对状态取平均；不对候选槽位加权。动作主指标只纳入可解状态，地图一致性则匹配其全部参考决策轮的分母。最优候选有并列时，随机一致率计入所有并列项。

reference 贪心采用原物理示范在该状态选中的动作，保留原并列处理；完整任务贪心采用原示范的到达与最短路计数。贪心不是所有读图模型的严格上限，随机也不是严格下限；原实验没有完整随机 rollout，因此任务图不画随机成功率，不能把一步 chance 外推成完整任务概率。读图一致性提高也不能单独证明地图的因果收益，须结合报告中的 Q 倒序干预。

## 重绘与导出

从仓库根目录运行（绘图仅需 matplotlib，已保存摘要足够；本次使用 matplotlib 3.10.6）：

```bash
python -m pip install matplotlib
python -m experiments.flamingo_map_reader.src.plot_learning_curves
```

如需从 Git 外原运行重算参照（使用项目环境与裁判，不运行模型推理）：

```bash
python -m experiments.flamingo_map_reader.src.collect_plot_baselines \
  --run-root RUN \
  --out experiments/flamingo_map_reader/results/learning_curve_baselines.json
```

每图同时保存 PNG（报告预览）、PDF（矢量排版）和 SVG（可编辑文字），文件同名。报告引用图像文件，保留原表格以核对细项；图与表均采用相同的成功定义。
