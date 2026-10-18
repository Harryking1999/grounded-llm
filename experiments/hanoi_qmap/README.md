# 汉诺塔 Qmap

用同一个纯 Q 范式，分别训练各盘数的状态地图。本任务独立维护 [数据](src/data.py)、[训练](src/run.py)、[模型适配](src/model.py)与[评测](src/evaluate.py)，不再使用混合 puzzle 训练入口。与其他任务共享基础 MLP、距离和排序方法，见[统一方法](../../tasks/QMAP_METHOD.md)。完整到达与六盘失败见[结果报告](results/report.md)，文字、数据和选定权重见[任务工作区](../../tasks/hanoi/README.md)。这不是一个模型跨盘数泛化；八盘已扩展完整精确距离及同轨道划分，新增配置为 [hanoi8.json](configs/hanoi8.json)，本轮暂无训练结论。

## 规则与状态

三根柱 A/B/C；n 个盘从小到大编号。状态记录每个盘在哪根柱，输入为 3n 维独热。每根柱上的盘按大小唯一排列，因此共有 3ⁿ 个合法状态。四、五、六盘分别为 81、243、729 个状态，图直径分别为 15、31、63。

动作是六种有序柱间移动；只能移动源柱顶盘，且不能放到更小的盘上。非法动作抛出异常，批量后继中非法槽记为 −1。每次合法移动计一步。环境完整枚举状态和边，用 BFS 标注任意两状态的精确最短距离；七、八盘用 SciPy 的无权图最短路实现同一裁判，避免逐元素 Python BFS 的开销；Q 选择器不读取该距离表。

## 数据与对称性

三根柱同时重命名保持距离和动作关系。将无序状态对及其六种柱置换视为一个关系轨道，完整轨道不能跨训练／验证／测试。

四至六盘把所有合法一步关系轨道放入训练，其他关系按轨道留出。训练混合完整遍历和真距离均衡。它检验的是**已知状态、已知一步拓扑下的长程关系补全**；后继到目标的其他关系可能见过，不是未见状态评测。

状态图由三个少一盘子图构成，移动最大盘的边连接它们。过桥前要先把较小盘搬到第三柱，所以不能用逐盘是否在目标柱上的差异替代最短距离。

## 配置与使用

[三盘](configs/study.json)、[四盘](configs/hanoi4.json)、[五盘](configs/hanoi5.json)、[六盘](configs/hanoi6.json)各有独立配置。三盘是较早合同，不与后续盘数作匹配泛化比较。

```bash
python -m experiments.hanoi_qmap.src.run --config experiments/hanoi_qmap/configs/hanoi4.json --out runs/hanoi_qmap/RUN_ID --source-commit COMMIT --device cuda:0
python -m experiments.puzzle_qmap.src.solve --task hanoi4 --start 0 --goal 80 --checkpoint runs/hanoi_qmap/RUN_ID/best.pt
python -m unittest discover -s experiments/hanoi_qmap/tests -v
```

去掉 checkpoint 参数运行的是精确裁判解。历史运行产物已归档，命令中的 RUN_ID 指新的运行目录；本次清理未启动新训练。

## 八盘首轮

继续采用完整关系轨道划分与六种柱置换增强；全部状态已知，留出的是关系而不是状态。训练预算随完整关系数量增加，正式参数只在合同中维护。除随机题外，最终固定报告六个标准整塔方向的完整 rollout 和循环。依赖项目环境中的 SciPy。
