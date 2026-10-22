# 推箱子数据

生成入口：[data.py](../../../grounded_llm/tasks/sokoban/data.py)。参数和划分数量只维护在[配置](../../../experiments/new_tasks_qmap/configs/sokoban.json)。运行输出为忽略目录中的 data.npz、scenes.jsonl.gz 和 data_summary.json，包含场景、状态编码、全部合法动作后继、动作成本、精确距离、终止掩码、起点及划分。不可达距离是正无穷，训练代理标签另存，不覆盖真值。

按完整墙体布局划分，三个集合墙体不重复；旋转和镜像同构尚未额外剔除。每个场景枚举玩家位置和箱子组合，保留死锁；从精确可解的非终止状态固定抽取rollout起点。随机ASCII种子仅用于构造静态布局，不保证其摆箱可解；实际评测起点在data.npz。

[example.json](example.json)仅为结构与规则说明，不作为独立评测题。

## Boxoban 文字加难题

后续文字筛查使用[官方 Boxoban 固定提交](https://github.com/google-deepmind/boxoban-levels/blob/78011d8c86ad423bfb2e1005012393b4c3c24409/README.md)的不同题目，不复用首轮自生成小棋盘。三个子集规则均为10×10、四箱四目标，改变的是布局；难度名由上游给定，不是本项目按距离重新分档。

| 子集 | 上游筛选依据 | 本项目来源 |
|---|---|---|
| unfiltered | 按原生成过程产生，未按策略求解失败进一步筛选 | `unfiltered/test/000.txt` |
| medium | 较弱 DRC 策略尝试10次仍未解出，来自新生成题 | `medium/valid/000.txt` |
| hard | 在独立 medium 候选中，较强 DRC 策略尝试10次仍未解出的3332题 | `hard/000.txt` |

筛选过程见 [Guez等，ICML2019，附录§7](https://proceedings.mlr.press/v97/guez19a/guez19a-supp.pdf)。这些是相对于原作者策略的经验难度，不保证对Flash单调变难，也不意味着hard题的最短解一定更长。

本项目每档最多顺序测3题，首次有效完成失败即停止；先由有界推空间搜索找到合法参考解才接纳题目，因此还带有本地求解器筛选，不能作为完整benchmark成绩。参考轨迹是合法走／推动作的上界，不冒充最短走动距离。当前结果与首轮小棋盘分开，见[文字基线报告](../../../experiments/task_text_baselines/results/report.md)。
