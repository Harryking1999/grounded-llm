# Token Swapping：任务文字与监督格式

采用[顺序Token Swapping](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2016.66)，公开实例来源为[tket固定问题](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/Data/FixedCompleteSolutions.hpp)。每点恰有一个不同棋子，目标为题目指定的完整排列；归位是其中一种目标。一次只交换一条边的两端，不使用并行轮数、部分映射或颜色等价规则。完整计划与指定目标裁判已接入；Qmap已完成多组训练但未通过验收，逐轮LLM训练尚未开展。历史Flash筛查的归位提示冻结在原suite中。

## 共享任务描述与规则

```text
Find a valid sequence of swaps from the initial placement to the goal placement, with as few swaps as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer swaps.

Rules: The graph has {vertex_count} vertices, numbered from 0 to {max_vertex_id}. Each vertex holds exactly one token. Token labels are the same integers, each used exactly once.
In one action, choose two distinct vertices joined by an undirected edge and exchange the tokens on those two vertices.
Specify vertex IDs in each action, not token labels. The graph stays fixed. Tokens cannot jump along multiple edges in one action.
Swaps are sequential: one edge swap costs one move. Swaps cannot be combined into parallel rounds.
The goal is to match the specified goal placement exactly: for every vertex v, its token must equal entry v of the goal list. Minimize the total number of edge swaps.
Use at most {action_limit} actions.

Undirected edges (each edge listed once):
{edges}

Initial placement: list entry at index v is the token currently on vertex v.
{initial_tokens}

Goal placement, using the same list convention:
{goal_tokens}
```

## 地图组／CoT 组：每轮环境更新

两组获得相同真实状态与合法候选。地图另供Q向量；精确最优成本和参考解不进入文字。

```text
[Environment update]
Current tokens by vertex: {current_tokens}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: swap(vertices {vertex_a}, {vertex_b}); immediate cost: 1
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of arrays; each action contains the two distinct integer vertex IDs of an existing edge) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

完成、最优交换数与格式分别记录；多余交换计入成本，不能截去首次到达目标后的动作来获得最优分。

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Swapping vertices {vertex_a} and {vertex_b} is legal and preserves a minimum-swap route to the goal.
<action>{local_id}</action>
```

排序包含最近至多10个候选及`current`，使用`<`与`=`，由实际冻结Q生成。Reason须经裁判验证；尚未生成SFT数据，无地图组删除排序行。

直接完成：

```text
Reason: Swapping vertices {vertex_a} and {vertex_b} makes the placement match the specified goal.
<action>{local_id}</action>
```

已经完成：

```text
Reason: The placement already matches the specified goal; no further swap is needed.
<done/>
```

## 完整计划结构说明例

仅说明格式，不进入测试题提示。

```json
{"actions": [[0, 1]], "final_status": "solved"}
```
