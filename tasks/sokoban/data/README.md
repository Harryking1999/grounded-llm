# 推箱子数据

当前使用[DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels/blob/78011d8c86ad423bfb2e1005012393b4c3c24409/README.md)的hard题目：10×10、四箱四目标。每题保留原始布局，并明确指定箱子和玩家的最终位置。

## 难度等级是什么意思

三个等级来自同一个公开题库，棋盘大小和箱子数量相同，区别在于布局的筛选方式。

| 等级 | 上游定义 |
|---|---|
| unfiltered | 原生成过程得到的题目，未按策略失败进一步筛选 |
| medium | 较弱求解策略尝试10次仍未完成的题目 |
| hard | 在medium候选中，较强策略尝试10次仍未完成的3,332题 |

筛选方法见[原论文附录第7节](https://proceedings.mlr.press/v97/guez19a/guez19a-supp.pdf)。这是相对原作者策略的难度。

## 本项目使用的范围

当前已扩展到1,202个场景：1,000个训练、100个验证、100个测试，另有2个已知困难题。原来的64张训练及16张验证／测试棋盘均保留在各自划分中。只接纳能完整计算精确距离的布局，实际规模和同题比较见[覆盖量结果](../../../experiments/new_tasks_qmap/results/sokoban/sokoban_coverage_comparison.json)，选题与筛选条件见[配置](../../../experiments/new_tasks_qmap/configs/sokoban/sokoban_coverage1000_pairs.json)。

同一墙体及其旋转、镜像版本放在同一组。每个布局含多个完整目标，距离按玩家实际走动和推动的总步数计算。无解关系明确保留，不能把合法死路从候选中删除。

原题359、362已用于开发和DeepSeek诊断，单列记录，历史6×6两箱试验与当前hard数据不同，结果保留在[首轮记录](../../../experiments/new_tasks_qmap/results/pilot/summary.json)。[example.json](example.json)只是格式示例。

64张／1,000张训练棋盘的两组对照已完成，按棋盘均衡抽样；全部实际训练批次的距离与合法候选已核对。见[实验入口](../../../experiments/new_tasks_qmap/README.md#推箱子覆盖量对照)。
