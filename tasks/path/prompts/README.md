# 寻路：任务文字与监督格式

状态：本任务全部文本已确认。按用户统一规则，无地图组直接去掉地图排序，保留相同 Reason 与动作；不单独重复确认。

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

使用真实当前节点、实际走过的节点序列和本轮全部合法邻居；不因历史访问过滤候选。地图组额外获得当前／目标／合法后继 Q 槽位，目标与邻接表沿用首轮文本。

```text
[Environment update]
Current node: {current_node}
Actual executed path: [{visited_nodes_in_order}]
Legal next moves:
{local_id}: move({current_node},{next_node})
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## SOTA 组／32B 组：输出格式

共享同一任务描述与规则，按用户确认的格式一次输出完整剩余计划，无中途反馈。不把旧 32 节点 DAG 或旧 Q/V 条件移作新结果。

```text
Plan the complete remaining sequence from the current node and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with integer "from" and "to" fields) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

## 监督答案：排序／Reason／Action

### 地图组：已确认

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving from {from} to {to} is legal and leaves a route to the goal with one fewer move needed.
<action>{local_id}</action>
```

排序沿用积木：最近的至多 10 个候选加 `current`，用 `<`、`=` 表示远近与并列。地图排序来自选定 Q，环境裁判另外验证动作与依据，不把地图排序自动视为正确动作标签。

动作直接到达目标时，Reason 改为：

```text
Reason: Moving from {from} to {to} is legal and reaches the goal node.
```

已在目标时：

```text
Reason: The current node is the goal; no further move is needed.
<done/>
```

### 无地图 CoT 组

使用上面相同的 Reason、到达句式和停止答案，只去掉地图排序行，即 R→A。
