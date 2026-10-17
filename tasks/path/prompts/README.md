# 寻路：任务文字与监督格式

状态：共享任务描述与规则已由用户确认；每轮环境更新、各组输出格式与监督答案仍待逐项确认。

## 共享任务描述与规则

各组使用以下同一段描述与规则。按用户决定删除允许重访的说明，不添加禁止重访的限制，也不改变环境合法动作。

```text
Find a valid path from the initial node to the goal node, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The graph is undirected and unweighted. Nodes are numbered 0 through {max_node_id}.
An action move(from,to) moves from the current node to an adjacent node along a listed edge.
Each move costs one step.
Reach the goal node exactly.

Neighbors:
{adjacency_list}

Initial node: {initial_node}
Goal node: {goal_node}
```

## 地图组／CoT 组：每轮环境更新

待填：真实当前节点、目标、实际路径、本轮候选 ID 与下一节点的对应表。地图组额外获得当前／目标／合法后继 Q 槽位。

```text
{environment_update_to_be_discussed}
```

## SOTA 组／32B 组：输出格式

共享同一任务描述与规则；完整路径 JSON、终止状态及反馈方式待确定，不把旧 32 节点 DAG 或旧 Q/V 条件移作新结果。

```text
{sota_32b_output_contract_to_be_discussed}
```

## 监督答案：排序／Reason／Action

待填地图排序、CoT／短 Reason、动作与终止答案，并说明哪些组监督哪些字段。不从地图自动正确这一假设生成标签；标签仍由离线环境裁判提供。
