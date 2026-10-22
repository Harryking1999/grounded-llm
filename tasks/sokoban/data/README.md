# 推箱子数据

生成入口：[data.py](../../../grounded_llm/tasks/sokoban/data.py)。参数和划分数量只维护在[配置](../../../experiments/new_tasks_qmap/configs/sokoban.json)。运行输出为忽略目录中的 data.npz、scenes.jsonl.gz 和 data_summary.json，包含场景、状态编码、全部合法动作后继、动作成本、精确距离、终止掩码、起点及划分。不可达距离是正无穷，训练代理标签另存，不覆盖真值。

按完整墙体布局划分，三个集合墙体不重复；旋转和镜像同构尚未额外剔除。每个场景枚举玩家位置和箱子组合，保留死锁；从精确可解的非终止状态固定抽取rollout起点。随机ASCII种子仅用于构造静态布局，不保证其摆箱可解；实际评测起点在data.npz。

[example.json](example.json)仅为结构与规则说明，不作为独立评测题。
