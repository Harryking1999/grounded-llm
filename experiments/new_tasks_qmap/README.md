# 三个新任务的固定目标 Qmap 可行性

检验手电过桥的加权成本、推箱子的不可逆死路、24点的算术可解性，能否被现有状态 MLP 和距离／排序监督表示。每次调用只处理一个任务；三个模型、数据和结果独立，不共用权重，也不改动已有任务训练器。

| 任务 | 规则／数据入口 | 配置 |
|---|---|---|
| 手电过桥 | [任务](../../tasks/bridge/README.md) | [bridge.json](configs/bridge.json) |
| 推箱子 | [任务](../../tasks/sokoban/README.md) | [sokoban.json](configs/sokoban.json) |
| 24点 | [任务](../../tasks/game24/README.md) | [game24.json](configs/game24.json) |

## 运行

```bash
python -m experiments.new_tasks_qmap.src.run --config experiments/new_tasks_qmap/configs/bridge.json --out runs/new_tasks/bridge --source-commit COMMIT --device cuda:0
```

另两任务替换配置和输出目录。使用35016已确认空闲GPU，沿用平台项目Python；启动前复查占用。执行有限配置预算后退出，不追加实验。源码以已提交归档传输；远端使用节点临时盘运行，运行结束回存持久盘。具体路径与PID只写本地忽略的 `runs/new_tasks_qmap/runtime.json`。

输出包括原始图、来源题目、场景索引、初始模型指标、训练日志、best/last完整状态、逐状态预测和最终摘要。checkpoint存储优化器和采样状态，但本入口未实现恢复参数；不能把best权重称为已支持精确续训。已有目录拒绝覆盖。

## 比较与解释

编码器不单独拼接查询目标，但推箱子的输入通道含目标格，因此当前向量依赖目标布局。本轮只监督同场景既定目标的距离，不是任意目标地图验证。推箱子目标是集合，使用目标条件参照向量。不可达仍用真值正无穷保存，Huber与soft-rank使用配置中的有限代理值；不能声称有限向量表示了无穷距离。过桥按即时耗时＋后继距离评分，其余两任务即时成本为1。

每个任务报告训练场景、验证场景及留出场景的有限距离误差、最短动作正确率、严格候选排序、选入死路比例、整题成功与最优数。验证成功率及动作率选checkpoint；测试只作固定初始和最终读数，不参与选择。所有策略共享合法候选、起点和固定步数上限。oracle使用精确距离；简单启发式分别是尚未过桥者的最长耗时、最小箱子目标曼哈顿匹配、离24最近的当前数值；额外保留只看即时成本及未训练模型。没有搜索、死路屏蔽、反悔或额外推理预算。

数据公开源见各任务README；自行生成的过桥和小推箱子场景不宣称是标准benchmark。重复训练曝光保存在运行结果中；24点初题划分不保证中间状态独立。单种子、有限预算的首轮结果只用于定位可行性，不能替代正式模型验收、LLM读图增益或跨任务迁移。

验证：`python -m unittest discover -s experiments/new_tasks_qmap/tests -v`；文字同步：`python -m tasks.export_texts --check`。结论和数值见[报告](results/report.md)。

失败定位：`python -m experiments.new_tasks_qmap.src.diagnose --root runs/new_tasks_qmap/pilot_27893ea --out tmp/new_tasks_diagnosis.json`。该入口只读取保存的数据与预测，不训练或重新选择权重。
