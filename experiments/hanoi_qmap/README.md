# 三层汉诺塔 Qmap

研究问题：在全部合法状态上，用部分精确关系训练目标无关 Q，能否补全剩余最短步数，并通过真实合法候选的 Q 距离完成任意起终点任务？模型与评测分工见 [共享核心](../puzzle_qmap/README.md)，正式参数只维护在 [configs/study.json](configs/study.json)。首轮结果记录在 [results/report.md](results/report.md)。

## 状态、动作、合法性与求解

三根柱 A/B/C，三个盘编号 1/2/3，1 最小。状态为每个盘所在柱的三元组，按盘从小到大排列，柱编号 0/1/2；紧凑编号为三进制编码。每根柱上的盘按大小唯一排序，因此三元组的全部组合都是合法状态，不另枚举非法堆叠。`state_id` 验证输入，`describe` 返回每柱从下到上的盘序列。

动作是有序柱对，只有源柱顶盘存在且小于目标柱顶盘时合法。无效动作抛出异常；批量 `successors` 保留六个动作槽，不合法槽为 -1。每次移动一个顶盘计一步。环境完整枚举合法状态和转移，再对每个目标 BFS，给任意状态对精确最短距离；`solve` 沿距离下降返回可重放最短动作。没有把递归算法藏在 Q 的选择器中。

Q 输入是三个盘所在柱的 one-hot；直接复用积木共享编码器，与目标无关。数据按无序状态对的六种柱置换轨道划分；每个训练轨道取一个种子关系，再同时重命名两端，完整增强并去重，恒等对加入训练。所有对称等价关系留在同一划分。测试是**同一批状态的留出关系轨道补全**，不是未见盘数或新状态泛化。训练候选的每个后继—目标关系都必须在训练轨道内。详细变换与损失见共享核心；没有动作条件转移头。

## 运行

```bash
python -m unittest discover -s experiments/hanoi_qmap/tests -v
python -m unittest discover -s experiments/puzzle_qmap/tests -v
python -m experiments.puzzle_qmap.src.run --config experiments/hanoi_qmap/configs/study.json --out runs/hanoi_qmap/RUN_ID --source-commit COMMIT --device cuda:0
python -m experiments.puzzle_qmap.src.solve --task hanoi3 --start 0 --goal 26
python -m experiments.puzzle_qmap.src.solve --task hanoi3 --start 0 --goal 26 --checkpoint runs/hanoi_qmap/RUN_ID/best.pt
```

测试穷举全部起终点的最短解重放、全部合法／非法动作、反向边和训练监督划分。精确求解不需要 torch；纯 Q 不需要加载距离表。
