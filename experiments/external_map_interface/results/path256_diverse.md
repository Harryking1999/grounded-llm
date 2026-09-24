# 寻路实验：显式候选地图距离

## 目的与做法

我们想看：不训练 LLM 理解 Q token，而是在每一步直接告诉它认知地图预测的候选动作距离，能否帮助它寻路。

实验使用两张独立的 256 节点图，每张图选 16 道起终点题、每题运行 8 次。两组都使用 **Qwen3-4B 非 thinking**，逐步看到当前节点、目标、已经走过的路径、图的邻接关系和合法动作；执行动作后，环境反馈实际到达的节点。主对照组不看地图距离；地图组只多看每个候选动作的 learned Q-map 距离。这里的距离由训练出的 Q/V 计算，**不是真实最短路距离**。

## 结果：两批各 128 条，合并比较

| 条件 | 第一张图到达 /128 | 第二张图到达 /128 | 合计到达 /256 | 合计最短路 /256 |
| --- | ---: | ---: | ---: | ---: |
| 原始 Qwen3-4B 非 thinking | 69 | 53 | **122** | 4 |
| 加入候选地图距离 | 90 | 97 | **187** | 12 |

合计到达率从 **47.7% 提高到 73.0%**，增加 65/256，即 **25.4 个百分点**。两张图都观察到了到达目标的提升。最短路总数从 4 增至 12，但增加的 8 条都来自第一张图；第二张图两组同为 4 条。因此目前最明确的结果是：**显式候选地图距离帮助非 thinking 模型到达目标**，最短路的改善还没有在第二张图上重复出现。

## 模型每一步看到什么

下面是地图组 prompt 中与当前状态和候选动作有关的片段；完整 prompt 还包含图的邻接表。无地图组保留相同信息，只去掉每个动作后的地图距离。

```text
Current node: 4
Goal node: 98
Executed node sequence: 4
Legal actions (already visited nodes excluded):
action_id=26, to_node=216, learned_map_distance_to_goal=2.415296
action_id=28, to_node=251, learned_map_distance_to_goal=2.593553
action_id=24, to_node=120, learned_map_distance_to_goal=2.612124

Choose ONE action now. The environment executes it and provides the actual new state for the next decision.
Respond with one JSON object: {"action_id": integer}.
```

地图在这里给出的是各动作执行后预测状态到目标的距离。模型选择一个动作，环境执行后更新当前节点，再生成下一步的候选信息。

## 目前遇到的问题

**Thinking 模型**在第一张图的逐步实验中，经常在给出单步动作前用完输出预算：无地图组 117/128 条、地图组 124/128 条轨迹发生截断。**Qwen3-4B-Instruct-2507** 在第二张图上也有截断（无地图 26/128、地图 19/128）；更主要的问题是它常输出长篇推导，而不是要求的单个 JSON 动作，两组分别有 102/128、109/128 条因此失败。Instruct 两组目前都没有完成路径。

所以现有结果能清楚比较的是非 thinking Qwen3-4B 的两组。Thinking 与 Instruct 需要先解决单步回答过长、动作格式不稳定的问题，才能直观比较地图距离带来的变化。
