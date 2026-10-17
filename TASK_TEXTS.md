# 各任务文字、提示与监督答案总览

本文件由 `python -m tasks.export_texts` 生成。原文维护在 `tasks/<task>/prompts/README.md`，示例维护在对应 `data/example.json`；积木逐轮文字示例由其准备脚本生成。项目进展见 [状态页](docs/PROJECT_STATUS_AND_TODO.md)。


## 积木

来源：`tasks/blocks/prompts/README.md`

### 积木：任务文字与监督格式

状态：现有八形状、10×10 棋盘 SFT 条件已填入。这里的“八形状”是动作目录，不等于每个生成棋盘恰好含八次操作；新的 12 积木条件需另定配置。

#### 共享任务描述与规则

以下文字沿用当前 `blocks_prompt.initial_prompt`；初始／目标棋盘由实例填入。

```text
Find a valid sequence of removals from the initial board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 10x10 binary grid. 1 means occupied and 0 means empty.
An action remove(shape_id,row,col) removes one listed shape at its zero-based top-left anchor.
Every occupied shape cell must overlap a currently occupied board cell and lie inside the board.
Only those cells are removed. Empty shape cells impose no constraint.
Use only the listed orientations. Shapes may be reused; there is no gravity or inventory limit.
Reach the goal board exactly, including any occupied cells that must remain.
Shapes (/ separates rows): 0=11/10; 1=10/11; 2=11/01; 3=01/11; 4=1/1; 5=11; 6=1/1/1; 7=111

Initial board:
{initial_board_10_rows}

Goal board:
{goal_board_10_rows}
```

#### 地图组／CoT 组：每轮环境更新

```text
[Environment update]
Current board:
{current_board_10_rows}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: remove({shape_id},{row},{col})
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

使用真实执行结果更新，不能把模型自报棋盘当真值。本轮 ID 不是全局动作编号。地图组另收当前／目标／全部合法后继的冻结 Q 槽位；无地图组不收 Q。

#### 输出格式与 CoT 监督答案

当前地图组使用 ranking-first：先输出地图距离排序，再输出简短 Reason，最后输出一个控制标签。

```text
Map-distance ranking to the goal, closest to farthest: {nearest_10_candidates_plus_current_with_<and_=}.
Reason: Clearing {cells} is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
<action>{local_candidate_id}</action>
```

若本次动作直接到达目标，Reason 为 `Reason: Clearing {cells} is legal and leaves exactly the goal board.`；已在目标时为：

```text
Reason: The current board exactly matches the goal; no further removal is needed.
<done/>
```

当前无地图 7B 监督为相同事实依据与动作，即 R→A，不输出排序。它使用简短 Reason，并非长 CoT。后续长 CoT 文本、是否改为排序＋A，均待讨论，未替换已有运行。排序是 Q 距离顺序，动作正确性由环境裁判另判。

完整实际首轮格式指令与后续更新见同目录 `example.md`，由 `python -m tasks.blocks.data.prepare_example` 调用当前运行的提示函数生成。权威实现：`experiments/flamingo_map_reader/src/blocks_prompt.py`、`experiments/map_sft_baselines/src/common.py`、`experiments/llm_task_sft/src/protocol.py`。

#### SOTA 组／32B 组

采用上面的共享任务描述与规则。按用户确定的完整计划条件，一次输出剩余动作，无中途环境反馈：

```text
Plan the complete remaining sequence from the current board and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with integer shape_id, row, col) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

动作使用物理参数而非本轮候选 ID；不要求 `board_after`。已在目标时使用 `{"actions":[],"final_status":"solved"}`。输出预算待正式配置确认。与地图／CoT 逐轮条件的反馈方式不同，报告时注明。旧 SOTA／32B 整段计划条件还使用过不同形状目录与空目标，不能直接移用为本轮匹配结果。


### 数据说明例

```json
{
  "example_kind": "constructed_schema_fixture_not_training_sample",
  "task": "blocks",
  "start": "25769803779",
  "goal": "25769803776",
  "board_row": null,
  "shortest_moves": 1,
  "route_ids": [
    1
  ],
  "pairs": [
    [
      "25769803779",
      "25769803776"
    ]
  ],
  "initial_board": [
    "1100000000",
    "0000000000",
    "0000000000",
    "0001100000",
    "0000000000",
    "0000000000",
    "0000000000",
    "0000000000",
    "0000000000",
    "0000000000"
  ],
  "goal_board": [
    "0000000000",
    "0000000000",
    "0000000000",
    "0001100000",
    "0000000000",
    "0000000000",
    "0000000000",
    "0000000000",
    "0000000000",
    "0000000000"
  ],
  "reference_action": {
    "shape_id": 5,
    "row": 0,
    "col": 0
  },
  "sota_answer": {
    "actions": [
      {
        "shape_id": 5,
        "row": 0,
        "col": 0
      }
    ],
    "final_status": "solved"
  },
  "cot_ra_answer": "Reason: Clearing (0,0) and (0,1) is legal and leaves exactly the goal board.\n<action>1</action>",
  "map_answer_template": "{ranking_from_selected_Q}\nReason: Clearing (0,0) and (0,1) is legal and leaves exactly the goal board.\n<action>1</action>"
}
```


### 已渲染文字示例

说明：这是构造的单步可解格式例，不是训练集抽样，也不是评测结果。排序须由选定 Q 生成，示例不伪造排序标签。

## 地图组首轮

```text
Find a valid sequence of removals from the initial board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 10x10 binary grid. 1 means occupied and 0 means empty.
An action remove(shape_id,row,col) removes one listed shape at its zero-based top-left anchor.
Every occupied shape cell must overlap a currently occupied board cell and lie inside the board.
Only those cells are removed. Empty shape cells impose no constraint.
Use only the listed orientations. Shapes may be reused; there is no gravity or inventory limit.
Reach the goal board exactly, including any occupied cells that must remain.
Shapes (/ separates rows): 0=11/10; 1=10/11; 2=11/01; 3=01/11; 4=1/1; 5=11; 6=1/1/1; 7=111

Initial board:
1100000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000

Goal board:
0000000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000

Response format:
First output the map-distance ranking. Next output a short factual line beginning with "Reason:". Finally output the action line.
For a nonterminal board, choose a legal candidate; state the cells removed, whether goal cells are preserved, and how the removal advances a valid completion.
Use: Reason: Clearing CELLS is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
If that action reaches the goal, use: Reason: Clearing CELLS is legal and leaves exactly the goal board.
At the goal, output: Reason: The current board exactly matches the goal; no further removal is needed.
The final line is <action>ID</action> or, at the goal, <done/>.
Before the action line, output "Map-distance ranking to the goal, closest to farthest: RANKING." Rank the nearest 10 candidates plus current using < and =; input contains every legal candidate.

[Environment update]
Current board:
1100000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
Actual executed actions: (none)
Legal next moves:
1: remove(5,0,0)
2: remove(5,3,3)
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## 无地图 R＋A 组首轮

```text
Find a valid sequence of removals from the initial board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 10x10 binary grid. 1 means occupied and 0 means empty.
An action remove(shape_id,row,col) removes one listed shape at its zero-based top-left anchor.
Every occupied shape cell must overlap a currently occupied board cell and lie inside the board.
Only those cells are removed. Empty shape cells impose no constraint.
Use only the listed orientations. Shapes may be reused; there is no gravity or inventory limit.
Reach the goal board exactly, including any occupied cells that must remain.
Shapes (/ separates rows): 0=11/10; 1=10/11; 2=11/01; 3=01/11; 4=1/1; 5=11; 6=1/1/1; 7=111

Initial board:
1100000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000

Goal board:
0000000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000

Response format:
Output a short factual line beginning with "Reason:", then the requested lines.
For a nonterminal board, choose a legal candidate; state the cells removed, whether goal cells are preserved, and how the removal advances a valid completion.
Use: Reason: Clearing CELLS is legal, preserves all goal cells, and leaves a solvable board with one fewer removal needed.
If that action reaches the goal, use: Reason: Clearing CELLS is legal and leaves exactly the goal board.
At the goal, output: Reason: The current board exactly matches the goal; no further removal is needed.
The final line is <action>ID</action> or, at the goal, <done/>.
Do not output candidate rankings.

[Environment update]
Current board:
1100000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
Actual executed actions: (none)
Legal next moves:
1: remove(5,0,0)
2: remove(5,3,3)
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## 执行后的下一轮更新

```text
[Environment update]
Current board:
0000000000
0000000000
0000000000
0001100000
0000000000
0000000000
0000000000
0000000000
0000000000
0000000000
Actual executed actions: remove(5,0,0)
Legal next moves:
1: remove(5,3,3)
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

## 对应监督答案

```text
Reason: Clearing (0,0) and (0,1) is legal and leaves exactly the goal board.
<action>1</action>
```

已到达后的停止答案：

```text
Reason: The current board exactly matches the goal; no further removal is needed.
<done/>
```


## 寻路

来源：`tasks/path/prompts/README.md`

### 寻路：任务文字与监督格式

状态：本任务全部文本已确认。按用户统一规则，无地图组直接去掉地图排序，保留相同 Reason 与动作；不单独重复确认。

#### 共享任务描述与规则

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

#### 地图组／CoT 组：每轮环境更新

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

#### SOTA 组／32B 组：输出格式

共享同一任务描述与规则，按用户确认的格式一次输出完整剩余计划，无中途反馈。不把旧 32 节点 DAG 或旧 Q/V 条件移作新结果。

```text
Plan the complete remaining sequence from the current node and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with integer "from" and "to" fields) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

#### 监督答案：排序／Reason／Action

##### 地图组：已确认

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

##### 无地图 CoT 组

使用上面相同的 Reason、到达句式和停止答案，只去掉地图排序行，即 R→A。


### 数据说明例

```json
{
  "example_kind": "schema_fixture_not_training_sample",
  "task": "path",
  "neighbors": {
    "0": [
      1
    ],
    "1": [
      0,
      2
    ],
    "2": [
      1
    ]
  },
  "current": 0,
  "goal": 2,
  "candidates": [
    {
      "id": 1,
      "action": {
        "from_node": 0,
        "to_node": 1
      },
      "successor": 1
    }
  ],
  "qmap_supervision": {
    "pair": {
      "source": 0,
      "target": 2,
      "distance": 2
    },
    "decision": {
      "parent": 0,
      "goal": 2,
      "child_distances": [
        1
      ]
    }
  },
  "supervised_answer": {
    "ranking": null,
    "reason": null,
    "action": null
  }
}
```


## 汉诺塔

来源：`tasks/hanoi/prompts/README.md`

### 汉诺塔：任务文字与监督格式

状态：整套文本已由用户确认，并按要求在动作中加入盘编号。八盘是新一轮候选，文字示例不代表八盘环境、训练或权重已经完成。

#### 共享任务描述与规则

```text
Find a valid sequence of moves from the initial configuration to the goal configuration, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: There are three pegs, A, B, and C, and {disk_count} disks numbered from 1 (smallest) to {disk_count} (largest).
Each peg lists its disks from bottom to top.
An action move(disk_id,from,to) moves the specified disk from one peg to another. The disk must be on top of the source peg.
Only one disk may be moved at a time. Never place a larger disk on a smaller disk.
Each move costs one step.
Reach the goal configuration exactly.

Initial configuration:
{initial_configuration}

Goal configuration:
{goal_configuration}
```

`configuration` 的显示格式如下；数字是固定盘编号，最右侧是柱顶，`[]` 表示空柱。八盘整塔初始摆放的说明例：

```text
A: [8, 7, 6, 5, 4, 3, 2, 1]
B: []
C: []
```

例如执行 `move(1,A,C)` 后：

```text
A: [8, 7, 6, 5, 4, 3, 2]
B: []
C: [1]
```

初始、当前和目标摆放均用相同格式，可表达任意合法目标，不只整塔。盘编号与本轮候选编号相互独立；加入盘编号不改变合法动作集合，解析时须校验它确为源柱顶盘。

#### 地图组／CoT 组：每轮环境更新

地图组额外获得当前／目标／合法后继 Q 槽位；两组文字更新相同。

```text
[Environment update]
Current configuration:
{current_configuration}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: move({disk_id},{from},{to})
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current configuration and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with an integer "disk_id" and "from" and "to" fields, each containing "A", "B", or "C") and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

#### 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving disk {disk_id} from {from} to {to} is legal and leaves a solvable configuration with one fewer move needed.
<action>{local_id}</action>
```

排序规则沿用积木，最近的至多 10 个候选加 `current`，用 `<`、`=` 表示远近与并列。无地图组直接去掉排序行，保留相同 Reason 与控制标签。

动作直接到达目标时，Reason 改为：

```text
Reason: Moving disk {disk_id} from {from} to {to} is legal and reaches the goal configuration.
```

已在目标时：

```text
Reason: The current configuration exactly matches the goal; no further move is needed.
<done/>
```


### 数据说明例

```json
{
  "example_kind": "historical_three_disk_schema_fixture_not_eight_disk_training",
  "task": "hanoi3",
  "state_encoding": "smallest_disk_first_peg_indices",
  "current": [
    0,
    0,
    0
  ],
  "goal": [
    2,
    2,
    2
  ],
  "candidates": [
    {
      "id": 1,
      "action": "A>B",
      "successor": [
        1,
        0,
        0
      ]
    },
    {
      "id": 2,
      "action": "A>C",
      "successor": [
        2,
        0,
        0
      ]
    }
  ],
  "qmap_supervision": {
    "pair_distance": 7,
    "candidate_distances": [
      7,
      6
    ]
  },
  "supervised_answer": {
    "ranking": null,
    "reason": null,
    "action": null
  }
}
```


## 二阶魔方

来源：`tasks/cube2/prompts/README.md`

### 二阶魔方：任务文字与监督格式

状态：文字先占位；动作和呈现须对应当前二阶 Qmap。

#### 共享任务描述与规则

当前地图使用 DBL 角块固定参考系、角块身份与朝向输入，动作是 U、U′、R、R′、F、F′ 六种四分之一转。180° 需要两步；不能混用旧九动作 HTM 条件。需讨论模型侧采用贴纸、角块还是其他文字呈现，并完整解释旋转方向、固定观察框架、目标和结束判定。

```text
{task_description_and_rules_to_be_discussed}
```

#### 地图组／CoT 组：每轮环境更新

待填：真实当前魔方、目标、实际转动历史、候选 ID 与转动的对应关系。地图组额外获得当前／目标／真实后继 Q 槽位。

```text
{environment_update_to_be_discussed}
```

#### SOTA 组／32B 组：输出格式

待填完整转动序列、终止状态和是否允许输出额外说明；共享同一规则与呈现。

```text
{sota_32b_output_contract_to_be_discussed}
```

#### 监督答案：排序／Reason／Action

待填排序、CoT／短 Reason、单步动作与停止。复原和任意目标条件分别准备；当前复原断点不等于任意目标已解决。


### 数据说明例

```json
{
  "example_kind": "schema_placeholder_not_training_sample",
  "task": "cube2_qtm",
  "frame": "DBL_fixed_untwisted",
  "actions": [
    "U",
    "U'",
    "R",
    "R'",
    "F",
    "F'"
  ],
  "current": null,
  "goal": null,
  "candidates": [],
  "qmap_supervision": {
    "pair_distance": null,
    "candidate_distances": []
  },
  "supervised_answer": {
    "ranking": null,
    "reason": null,
    "action": null
  }
}
```
