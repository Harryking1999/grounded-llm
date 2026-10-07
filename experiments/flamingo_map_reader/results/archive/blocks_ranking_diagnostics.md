# 上一轮积木 final 的排序评分诊断

这部分来自 2026-10-06 的离线分析：`long_f73b700_20261004`，积木 epoch 5（step 245000），validation reference／rollout 各 1,100 题、编号 variant 0。它评价输出排序与地图距离排序的接近程度，不属于当前融合 FFN 实验，也不等同于动作可达或任务成功。

原临时脚本的三种口径已合并到一个[可复用评分入口](../../src/analyze_blocks_ranking.py)，计数与定义见[归档摘要](blocks_ranking_summary.json)。逐轮明细、原报告与图留在 Git 外的 `runs/imported/blocks_ranking/`。

| gain（候选相关性） | 位次折扣 | rollout 均分 | reference 均分 |
|---|---|---:|---:|
| 1/(1+地图距离) | 1/log2(i+1) | 90.46% | 95.09% |
| 地图名次等级：Top-K 为 K 至 1，其他为 0 | 1/log2(i+1) | 70.91% | 79.75% |
| 同上 | 1/i | 68.24% | 78.29% |

各口径都用理想排序得分归一化；真实距离并列共享名次等级，预测并列取组内平均 gain。`current` 不参与候选排名，缺失位置按零计，格式错误计零；无候选轮次无定义、排除并单列。全轮口径含作答前已不可达状态，rollout 计分 8,299 轮、reference 8,242 轮，另保留可解子集。不能把这三组百分比混成一个学习趋势，也不能以高分替代任务指标。

重算上一轮已保存案例（不运行 LLM 推理）：

```bash
python -m experiments.flamingo_map_reader.src.analyze_blocks_ranking \
  --run-root "$OLD_RUN_DIR" --out "$ANALYSIS_DIR" \
  --gain rank_grade --discount reciprocal_rank
```

入口也支持原距离 gain 与对数折扣；归档 JSON 中的 `ndcg` 字段沿用历史命名，实际公式以 `definitions` 为准。独立的 [diagnose_blocks_ranking.py](../../src/diagnose_blocks_ranking.py) 复用同一评分函数，分析原距离口径的排序分数与首步动作可达性；它是离线诊断，不参与被评测 agent 选动作。
