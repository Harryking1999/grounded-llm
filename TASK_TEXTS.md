# 各任务文字、提示与监督答案总览

本文件由 `python -m tasks.export_texts` 生成。原文维护在 `tasks/<task>/prompts/README.md`，示例维护在对应 `data/example.json`；积木逐轮文字示例由其准备脚本生成。项目进展见 [状态页](docs/PROJECT_STATUS_AND_TODO.md)。


## 积木

来源：`tasks/blocks/prompts/README.md`

### 积木：任务文字与监督格式

状态：当前主条件采用八形状、10×10 棋盘，按 12 次放置构造；“八形状”是动作目录，主评测配置见 [sol_high_40.json](experiments/task_text_baselines/configs/sol_high_40.json)。15×15、20 块、含 S/Z 的加难条件已停止，仅保留为补充探索，不改变此处规则。

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

使用真实执行结果更新，本轮 ID 不是全局动作编号。地图组另收当前／目标／全部合法后继的冻结 Q 槽位；无地图组不收 Q。

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

当前无地图 7B 监督为相同事实依据与动作，即 R→A，不输出排序。它使用简短 Reason，后续长 CoT 文本、是否改为排序＋A，均待讨论，未替换已有运行。排序是 Q 距离顺序，动作正确性由环境裁判另判。

完整实际首轮格式指令与后续更新见同目录 `example.md`，由 `python -m tasks.blocks.data.prepare_example` 调用当前运行的提示函数生成。权威实现：`experiments/flamingo_map_reader/src/blocks_prompt.py`、`experiments/map_sft_baselines/src/common.py`、`experiments/llm_task_sft/src/protocol.py`。

#### SOTA 组／32B 组

采用上面的共享任务描述与规则。按用户确定的完整计划条件，一次输出剩余动作，无中途环境反馈：

```text
Plan the complete remaining sequence from the current board and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects with integer shape_id, row, col) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

动作使用物理参数而非本轮候选 ID；不要求 `board_after`。已在目标时使用 `{"actions":[],"final_status":"solved"}`。输出预算以对应实验正式配置为准。与地图／CoT 逐轮条件的反馈方式不同，报告时注明。旧 SOTA／32B 整段计划条件还使用过不同形状目录与空目标。


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

说明：这是构造的单步可解格式例，排序须由选定 Q 生成。

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

共享同一任务描述与规则，按用户确认的格式一次输出完整剩余计划，无中途反馈。

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

排序沿用积木：最近的至多 10 个候选加 `current`，用 `<`、`=` 表示远近与并列。地图排序来自选定 Q，环境裁判另外验证动作与依据。

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

状态：整套文本已由用户确认，动作包含盘编号。当前八盘环境、训练与选定Qmap已完成；正式评测的目标限定为指定柱上的整塔，起点可以是任意合法状态。权重与适用范围见[选用记录](tasks/hanoi/Qmap/selection.json)。

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
Return JSON with "actions" (an array of objects with integer "disk_id" and string "from" and "to" fields; peg names are "A", "B", or "C") and "final_status" ("solved", "no_solution", or "unsolved").
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

状态：用户已确定参考 XoT，采用六面 `2×2` 数字矩阵和动作前后示例作为 LLM 侧表示。共享规则、每轮更新、完整计划输出和监督答案已补齐，用户已授权使用当前文字开展纯文字基线。双向贴纸转换已实现并检查；Q 编码器继续使用角块身份与朝向的 192 维独热输入。

#### 共享任务描述与规则

表示参考 [XoT 公开提示词](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/main/xot_all_in_one/xot/prompter/prompts/prompts_cube.py)：六面分别列出两行数字，0–5 是颜色类别。采用其一种复原配色：U=0、R=1、F=2、D=3、L=4、B=5。初始、当前和目标状态统一使用此格式。

当前地图使用 DBL 角块固定参考系，动作保持 U、U′、R、R′、F、F′ 六种四分之一转；不引入 XoT 的半转单步或 11 步上限。目标沿用本项目的明确目标状态。

```text
Find a valid sequence of turns from the initial cube state to the goal cube state, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: This is a 2x2 cube with six faces: Upper (U), Front (F), Down (D), Left (L), Right (R), and Back (B).
Each face is shown as a 2x2 grid. Numbers 0 through 5 represent fixed sticker colors.
The cube's reference frame stays fixed; do not rotate the whole cube.
Allowed turns are U, U', R, R', F, and F'. A letter turns that face 90 degrees clockwise as viewed directly from outside that face; an apostrophe means counterclockwise.
Each quarter-turn costs one step. A 180-degree turn requires two quarter-turns.
Reach the goal cube state exactly.

Action examples: each example starts independently from the state below.
Do not apply these example turns in sequence.

[Before each example]
Upper:
0 0
0 0
Front:
2 2
2 2
Down:
3 3
3 3
Left:
4 4
4 4
Right:
1 1
1 1
Back:
5 5
5 5

[After U]
Upper:
0 0
0 0
Front:
1 1
2 2
Down:
3 3
3 3
Left:
2 2
4 4
Right:
5 5
1 1
Back:
4 4
5 5

[After U']
Upper:
0 0
0 0
Front:
4 4
2 2
Down:
3 3
3 3
Left:
5 5
4 4
Right:
2 2
1 1
Back:
1 1
5 5

[After R]
Upper:
0 2
0 2
Front:
2 3
2 3
Down:
3 5
3 5
Left:
4 4
4 4
Right:
1 1
1 1
Back:
0 5
0 5

[After R']
Upper:
0 5
0 5
Front:
2 0
2 0
Down:
3 2
3 2
Left:
4 4
4 4
Right:
1 1
1 1
Back:
3 5
3 5

[After F]
Upper:
0 0
4 4
Front:
2 2
2 2
Down:
1 1
3 3
Left:
4 3
4 3
Right:
0 1
0 1
Back:
5 5
5 5

[After F']
Upper:
0 0
1 1
Front:
2 2
2 2
Down:
4 4
3 3
Left:
4 0
4 0
Right:
3 1
3 1
Back:
5 5
5 5

Initial cube state:
{initial_cube_state}

Goal cube state:
{goal_cube_state}
```

六面矩阵与六种四分之一转示例参考 XoT；每个示例独立从同一复原状态出发，不是连续执行。教学示例在各组共用，不提供当前测试题的后继或解题轨迹。已核对这六个示例与当前角块环境的转移一致；[双向贴纸转换](experiments/task_text_baselines/src/cube2.py)已通过随机状态往返检查。DBL 固定是内部参考系约束，不增加模型动作。

#### 地图组／CoT 组：每轮环境更新

两组使用相同文字，初始规则与目标沿用首轮输入。当前状态使用上面的六面矩阵格式，转动历史来自环境实际执行结果。地图组额外获得当前／目标／六个真实后继 Q 槽位；无地图组不收 Q。

```text
[Environment update]
Current cube state:
{current_cube_state}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: {turn}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

`{turn}` 为 U、U'、R、R'、F、F' 之一。每轮列出六个候选；候选 ID 是本轮控制地址，不是固定的转动编号。执行后由环境更新真实状态，不采用模型自报的状态。

#### SOTA 组／32B 组：输出格式

共享同一任务规则、六面表示和固定动作示例。一次输出完整剩余计划，无中途反馈；动作使用转动名称，不使用本轮候选 ID。

```text
Plan the complete remaining sequence from the current cube state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return only JSON with "actions" (an array of strings, each one of "U", "U'", "R", "R'", "F", or "F'") and "final_status" ("solved", "no_solution", or "unsolved"). Do not include any explanation or commentary outside the JSON.
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

格式示例（不对应上面的具体魔方）：

```json
{"actions":["R","U'","F"],"final_status":"solved"}
```

已在目标时输出 `{"actions":[],"final_status":"solved"}`。是否到达由环境执行验证，此组与地图／CoT 的逐轮反馈条件不同，结果中注明。

#### 监督答案：排序／Reason／Action

地图组沿用积木的排序→简短 Reason→单步动作：

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Turning {turn} is legal and leaves a cube state with one fewer turn needed to reach the goal.
<action>{local_id}</action>
```

排序包含六个候选和 `current`，按 Q 到目标的距离由近及远排列，用 `<`、`=` 表示远近与并列。动作正确性及“剩余最短步数减少一步”的依据由环境真值另行验证。

动作直接到达目标时：

```text
Reason: Turning {turn} is legal and reaches the goal cube state.
<action>{local_id}</action>
```

已在目标时：

```text
Reason: The current cube state exactly matches the goal; no further turn is needed.
<done/>
```

无地图 CoT 组直接去掉排序行，保留相同 Reason、动作和停止标签。以上均为简短 R＋A 监督，复原和任意可达目标共用文字格式，但数据与结果分别报告。


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


## 8-Puzzle（数字华容道）

来源：`tasks/puzzle8/prompts/README.md`

### 8-Puzzle：任务文字与监督格式

状态：参考 [XoT 提示词](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/main/xot_all_in_one/xot/prompter/prompts/prompts_npuzzle.py)的 3×3 数字表示与空格移动方向，按用户要求不加入动作示例，并精简冗余说明。用户已授权使用当前文字开展纯文字基线；目标说明例沿用其左上角为空格的布局，不继承其九步题集／预算。

#### 共享任务描述与规则

```text
Find a valid sequence of moves from the initial board to the goal board, as short as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a 3x3 grid containing tiles numbered 1 through 8 and one blank, represented by 0.
Each number appears exactly once.
An action Up, Down, Left, or Right moves the blank one cell in that direction by swapping it with the adjacent tile.
Each move costs one step.
Reach the goal board exactly.

Initial board:
{initial_board_3_rows}

Goal board:
{goal_board_3_rows}
```

中文：通过移动空格，将初始棋盘变为目标棋盘；优先完成，再尽量少走。Up／Down／Left／Right 指空格移动方向，数字块相应反向滑入空格。规定方向可避免将“空格向左”误读成“数字块向左”。各组共用上述规则文字，不加入动作教学示例。

目标说明例与 XoT 对齐，初始／当前状态使用同样格式：

```text
0 1 2
3 4 5
6 7 8
```

文字支持显式给定目标，实例按冻结题集显示具体棋盘；目标说明例不代替Qmap的[采用范围](tasks/puzzle8/Qmap/selection.json)。九个数的合法排列不保证与目标连通，准备数据时必须区分合法棋盘与可达任务。默认成功率题集使用可达状态，不把不可达题混入同一分母；若评测无解识别，单独列条件与指标。

#### 地图组／CoT 组：每轮环境更新

两组文字相同，规则与目标沿用首轮输入。地图组额外获得当前／目标／全部合法真实后继的 Q 槽位，无地图组不收 Q。

```text
[Environment update]
Current board:
{current_board_3_rows}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: {direction}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

只列空格当前位置允许的 2–4 个动作。候选 ID 是本轮控制地址，不是方向的固定编号；执行后由环境更新真实棋盘。

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current board and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return only JSON with "actions" (an array of strings, each one of "Up", "Down", "Left", or "Right") and "final_status" ("solved", "no_solution", or "unsolved"). Do not include any explanation or commentary outside the JSON.
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

中文：一次输出完整剩余计划，动作使用方向名称，无中途反馈；必须精确达到给定目标。沿用积木的 JSON 与终止语义。

#### 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving the blank {direction} is legal and leaves a board with one fewer move needed to reach the goal.
<action>{local_id}</action>
```

排序包含本轮全部合法候选与 `current`，按 Q 到目标的距离由近及远排列，用 `<`、`=` 表示远近与并列。动作及“剩余最短步数减少一步”由环境真值另行验证。简短 Reason 中的方向使用小写 up／down／left／right；动作名称保持首字母大写。

直接到达目标时：

```text
Reason: Moving the blank {direction} is legal and reaches the goal board.
<action>{local_id}</action>
```

已在目标时：

```text
Reason: The current board exactly matches the goal; no further move is needed.
<done/>
```

无地图 CoT 组直接删除排序行，保留相同 Reason、动作和停止标签。这里只定义简短 R＋A。


### 数据说明例

```json
{
  "example_kind": "one_move_schema_fixture_not_training_data",
  "task": "puzzle8",
  "state_encoding": "3x3_rows_top_to_bottom_zero_is_blank",
  "current": [
    [
      1,
      0,
      2
    ],
    [
      3,
      4,
      5
    ],
    [
      6,
      7,
      8
    ]
  ],
  "goal": [
    [
      0,
      1,
      2
    ],
    [
      3,
      4,
      5
    ],
    [
      6,
      7,
      8
    ]
  ],
  "executed_actions": [],
  "candidates": [
    {
      "id": 0,
      "action": "Left",
      "successor": [
        [
          0,
          1,
          2
        ],
        [
          3,
          4,
          5
        ],
        [
          6,
          7,
          8
        ]
      ]
    },
    {
      "id": 1,
      "action": "Down",
      "successor": [
        [
          1,
          4,
          2
        ],
        [
          3,
          0,
          5
        ],
        [
          6,
          7,
          8
        ]
      ]
    },
    {
      "id": 2,
      "action": "Right",
      "successor": [
        [
          1,
          2,
          0
        ],
        [
          3,
          4,
          5
        ],
        [
          6,
          7,
          8
        ]
      ]
    }
  ],
  "qmap_supervision": {
    "pair_distance": 1,
    "candidate_distances": [
      0,
      2,
      2
    ]
  },
  "supervised_answer": {
    "ranking": null,
    "reason": "Moving the blank left is legal and reaches the goal board.",
    "action": "<action>0</action>"
  }
}
```


## 手电过桥

来源：`tasks/bridge/prompts/README.md`

### 手电过桥：任务文字与监督格式

规则采用 [Backhouse 与 Truong 的手电过桥问题](https://people.cs.nott.ac.uk/psarb2/papers/abstract.html)的容量2情形。当前六人时间组合是本项目诊断场景。

以下定义可复用的文字协议。完整计划的提示与裁判已接入；逐轮地图／CoT部分是协议模板，尚无本任务LLM训练或已采用Qmap权重。

#### 共享任务描述与规则

```text
Find a valid sequence of crossings from the initial configuration to the goal configuration, with as little total crossing time as you can.
Reaching the goal is the first priority; among valid solutions, prefer less total crossing time.

Rules: There are {people_count} people, numbered from 0 to {max_person_id}, two banks (left and right), and one torch.
Each person has a fixed crossing time. In one action, one or two distinct people cross together with the torch.
All travellers must start on the same bank as the torch. After crossing, the travellers and the torch are on the opposite bank.
A crossing costs the maximum crossing time among its travellers. The same rules apply in both directions.
Reach the goal with every person and the torch on the right bank.
Use at most {action_limit} actions.

Crossing times (person ID: time units):
{crossing_times}

Initial configuration:
Left bank: {initial_left}
Right bank: {initial_right}
Torch bank: {initial_torch}

Goal configuration:
Left bank: []
Right bank: {all_people}
Torch bank: right
```

过桥以总时间为成本，不能用最少动作替代。地图距离排序仍表示后继到目标的剩余成本；选动作需加上本次过桥耗时，不能直接把后继距离最小的动作作为标签。

#### 地图组／CoT 组：每轮环境更新

两组共享规则、真实当前状态、实际动作历史与全部合法动作。地图组额外获得当前／目标条件／合法真实后继的Q槽位；精确距离、参考解和死路标签不进入模型文字。

```text
[Environment update]
Current configuration:
Left bank: {current_left}
Right bank: {current_right}
Torch bank: {current_torch}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: cross({person_ids}); crossing time: {immediate_cost}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of arrays; each action contains one or two distinct integer person IDs) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

沿用既有完整计划条件：无工具和中途反馈，不要求逐步状态报告。完成、最优成本与格式分别记录。模型和token预算以实验配置为准。

#### 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Crossing with people {person_ids} is legal and preserves a minimum-total-time route to the goal.
<action>{local_id}</action>
```

排序包含最近的至多10个候选及`current`，用`<`、`=`表示远近和并列。排序由实际冻结Q生成，上述Reason仅在裁判已验证相应事实时作为监督模板；尚未生成本任务SFT数据。无地图CoT组只删除排序行，保留相同Reason和控制标签。

直接到达目标时：

```text
Reason: Crossing with people {person_ids} is legal and reaches the goal configuration.
<action>{local_id}</action>
```

已经完成时：

```text
Reason: The current configuration exactly matches the goal; no further crossing is needed.
<done/>
```

#### 完整计划结构说明例

下例对应本任务[data/example.json](tasks/bridge/data/example.json)，只说明格式。

```json
{
  "actions": [
    [
      0,
      1
    ],
    [
      0
    ],
    [
      2,
      3
    ],
    [
      1
    ],
    [
      0,
      1
    ]
  ],
  "final_status": "solved"
}
```


### 数据说明例

```json
{
  "times": [
    1,
    2,
    5,
    10
  ],
  "initial": {
    "right_people": [],
    "torch": "left"
  },
  "goal": "all_right",
  "actions": [
    [
      0,
      1
    ],
    [
      0
    ],
    [
      2,
      3
    ],
    [
      1
    ],
    [
      0,
      1
    ]
  ],
  "optimal_cost": 17
}
```


## 推箱子

来源：`tasks/sokoban/prompts/README.md`

### 推箱子：任务文字与监督格式

推动与行走规则参考 [gym-sokoban](https://github.com/mpSchrader/gym-sokoban/blob/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89/gym_sokoban/envs/sokoban_env.py)，ASCII参考 [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels/blob/78011d8c86ad423bfb2e1005012393b4c3c24409/README.md)。保留的新增条件为Boxoban hard原题：10×10、四箱四目标；历史6×6两箱仅为诊断。下方规则模板按实际棋盘尺寸渲染，行走／推动成本与输出合同不变，具体题集及预算由实验配置维护。

以下定义可复用的文字协议。完整计划的提示与裁判已接入；逐轮地图／CoT部分是协议模板，尚无本任务LLM训练或已采用Qmap权重。

新训练与评测每题指定一张完整成功目标棋盘，包含玩家终点，字段为`goal_board`。没有该字段的适配仅为历史题目回放兼容，不安排原规则对照。

#### 共享任务描述与规则

```text
{task_objective}
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board is a {board_height}x{board_width} grid with walls, floor, targets, boxes, and one player.
ASCII symbols: # = wall, space = floor, . = target, $ = box, @ = player, * = box on target, + = player on target.
An action U, D, L, or R moves the player one cell up, down, left, or right.
If the adjacent cell contains a box, the action pushes that box one cell in the same direction, only if the cell beyond is empty floor or an unoccupied target.
The player cannot pass through walls or boxes, pull a box, or push more than one box at a time.
Each legal walk or push costs one step. A blocked move is illegal.
{goal_rule}
Use at most {action_limit} actions.

Initial board:
{initial_board}{goal_board_section}
```

方向指玩家移动方向，行走与推动均为一步。合法但进入死路的动作仍需列为合法候选。指定目标必须同时匹配箱子与玩家，输入保留目标格。

#### 指定完整目标文字

上方`task_objective`与`goal_rule`依次取下方两行；规则原文在此维护，不在代码中另存。

```text
Find a valid sequence of moves that exactly matches the specified goal board, as short as you can.
The goal is to match the specified goal board exactly, including the player and all box positions. Boxes are indistinguishable.
```

#### 历史目标文字：仅回放

仅用于没有`goal_board`的历史题目兼容。

```text
Find a valid sequence of moves that puts every box on a target, as short as you can.
The goal is that every box is on a target; the player's final position does not matter.
```

#### 地图组／CoT 组：每轮环境更新

两组共享规则、真实当前状态、实际动作历史与全部合法动作。地图组额外获得当前／完整目标状态／合法真实后继的Q槽位；精确距离、参考解和死路标签不进入模型文字。指定目标棋盘在两组任务描述中相同。

```text
[Environment update]
Current board:
{current_board}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: {direction}
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of strings, each one of "U", "D", "L", or "R") and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

沿用既有完整计划条件：无工具和中途反馈，不要求逐步状态报告。完成、最优成本与格式分别记录。模型和token预算以实验配置为准。

#### 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Moving {direction} is legal and leaves a solvable board with one fewer move needed to reach the goal.
<action>{local_id}</action>
```

排序包含最近的至多10个候选及`current`，用`<`、`=`表示远近和并列。排序由实际冻结Q生成，上述Reason仅在裁判已验证相应事实时作为监督模板；尚未生成本任务SFT数据。无地图CoT组只删除排序行，保留相同Reason和控制标签。

直接到达目标时：

```text
Reason: Moving {direction} is legal and reaches the required goal.
<action>{local_id}</action>
```

已经完成时：

```text
Reason: The required goal is already reached; no further move is needed.
<done/>
```

#### 完整计划结构说明例

下例对应本任务[data/example.json](tasks/sokoban/data/example.json)，只说明格式。

```json
{
  "actions": [
    "R"
  ],
  "final_status": "solved"
}
```


### 数据说明例

```json
{
  "initial": [
    "#####",
    "#@$.#",
    "#####"
  ],
  "goal_board": [
    "#####",
    "# @*#",
    "#####"
  ],
  "actions": [
    "R"
  ],
  "success": true,
  "move_cost": 1,
  "role": "small format illustration only; retained experiments use10x10 four-box hard layouts"
}
```


## 24点

来源：`tasks/game24/prompts/README.md`

### 24点：任务文字与监督格式

参考 [XoT的两数合并规则](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_all_in_one/xot/prompter/prompts/prompts_game24.py)及[公开环境](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_mcts/game24/Game24Game.py)。保留题目及运算规则；完整计划采用本项目统一JSON，不引入上游few-shot、MCTS或多次求解。

以下定义可复用的文字协议。完整计划的提示与裁判已接入；逐轮地图／CoT部分是协议模板，尚无本任务LLM训练或已采用Qmap权重。

#### 共享任务描述与规则

```text
Find a valid sequence of arithmetic operations that uses all four initial numbers exactly once and produces 24.
Reaching the goal is the first priority.

Rules: At each step, choose two different slots in the current sorted list, combine their numbers with one binary operation, remove both operands, and insert the result.
Sort the remaining list in ascending numerical order after every step. Slot indices are zero-based and refer to this newly sorted list, not to the initial numbers.
An action is [i, op, j], where i < j. Let x be the number at slot i and y the number at slot j.
The allowed op strings are "+" (x+y), "-" (x-y), "r-" (y-x), "*" (x*y), "/" (x/y), and "r/" (y/x).
Division by zero is illegal. Negative, zero, and fractional intermediate values are allowed; use exact arithmetic.
Equal numbers occupy distinct slots. Do not concatenate numbers, use powers, add constants, or split one number into two.
After exactly three operations, the list must contain the single exact value 24.

Initial sorted list:
{initial_numbers}

Goal list:
24
```

列表中的分数用最简分数字符串显示，排序按实际数值。动作只引用当前槽位，不能把表达式当动作提交。全部可解状态到[24]的距离恒为剩余数字个数减1；这里主要检验可解分支。

#### 地图组／CoT 组：每轮环境更新

两组共享规则、真实当前状态、实际动作历史与全部合法动作。地图组额外获得当前／目标条件／合法真实后继的Q槽位；精确距离、参考解和死路标签不进入模型文字。

```text
[Environment update]
Current sorted list:
{current_numbers}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: [{i}, "{operation}", {j}]
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of exactly three arrays [i, op, j], using the current sorted slot indices and the allowed operation strings) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

沿用既有完整计划条件：无工具和中途反馈，不要求逐步状态报告。完成、最优成本与格式分别记录。模型和token预算以实验配置为准。

#### 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Combining the numbers in slots {i} and {j} with {operation} is legal and leaves a solvable list with one fewer operation needed.
<action>{local_id}</action>
```

排序包含最近的至多10个候选及`current`，用`<`、`=`表示远近和并列。排序由实际冻结Q生成，上述Reason仅在裁判已验证相应事实时作为监督模板；尚未生成本任务SFT数据。无地图CoT组只删除排序行，保留相同Reason和控制标签。

直接到达目标时：

```text
Reason: Combining the numbers in slots {i} and {j} with {operation} is legal and leaves the single exact value 24.
<action>{local_id}</action>
```

已经完成时：

```text
Reason: The current list contains only 24; no further operation is needed.
<done/>
```

#### 完整计划结构说明例

下例对应本任务[data/example.json](tasks/game24/data/example.json)，只说明格式。

```json
{
  "actions": [
    [
      0,
      "r/",
      2
    ],
    [
      0,
      "r-",
      1
    ],
    [
      0,
      "r/",
      1
    ]
  ],
  "final_status": "solved"
}
```


### 数据说明例

```json
{
  "numbers": [
    3,
    3,
    8,
    8
  ],
  "actions": [
    [
      0,
      "r/",
      2
    ],
    [
      0,
      "r-",
      1
    ],
    [
      0,
      "r/",
      1
    ]
  ],
  "states": [
    [
      "3",
      "3",
      "8",
      "8"
    ],
    [
      "8/3",
      "3",
      "8"
    ],
    [
      "1/3",
      "8"
    ],
    [
      "24"
    ]
  ],
  "expression": "8/(3-8/3)"
}
```


## Rush Hour（移车出库）

来源：`tasks/rush_hour/prompts/README.md`

### Rush Hour：任务文字与监督格式

规则对齐[Fogleman公开实现](https://github.com/fogleman/rush/tree/3e3b8396891d1802b30cb3129ae31fdedb319480)；当前使用无墙6×6原题。一次滑动任意合法距离计一个move，不能使用逐格step标签。完整计划已接入；Qmap已完成多组训练但未通过验收，逐轮LLM训练尚未开展。

新训练与评测每题指定一张A已到最右端的完整目标棋盘，所有车的位置均需匹配，字段为`goal_board`。未指定该字段的适配仅为历史题目回放兼容，不安排原规则对照。历史出口目标题与指定完整目标题的成绩分开记录。

#### 共享任务描述与规则

```text
Find a valid sequence of vehicle slides from the initial board to the goal, with as few moves as you can.
Reaching the goal is the first priority; among valid solutions, prefer fewer moves.

Rules: The board has 6 rows and 6 columns. Each uppercase letter identifies one vehicle. A dot is an empty cell; x is a fixed wall.
Each vehicle has a fixed length and orientation. It can slide only along its orientation, without rotation or sideways movement.
A horizontal vehicle can slide left or right; a vertical vehicle can slide up or down.
An action slides exactly one vehicle by a nonzero integer displacement. Positive means right for a horizontal vehicle or down for a vertical vehicle; negative means left or up.
Every traversed cell must be clear of other vehicles and walls. Vehicles must stay inside the board.
A slide of any legal distance costs exactly one move. Minimize the total number of moves, not the number of cells travelled.
{goal_rule}
Use at most {action_limit} actions.

Vehicle catalogue:
{vehicle_catalogue}

Initial board, top row first:
{initial_board}{goal_board_section}
```

#### 指定完整目标文字

上方`goal_rule`取下方原文；规则只在本页维护。

```text
The goal is to match the specified goal board exactly, including the positions of every vehicle. Do not move A outside the board or remove it.
```

#### 历史目标文字：仅回放

仅用于没有`goal_board`的历史题目兼容。

```text
The goal is to place the horizontal target vehicle A at the rightmost position in its row, occupying columns 4 and 5 (zero-based). Other vehicles may end anywhere legal. Do not move A outside the board or remove it.
```

#### 地图组／CoT 组：每轮环境更新

两组共享真实状态、指定的完整目标棋盘和合法动作；地图额外提供当前、完整目标状态及合法真实后继的向量。参考成本与解序列不进入文字。

```text
[Environment update]
Current board:
{current_board}
Actual executed actions: {executed_actions_or_none}
Legal next moves:
{local_id}: slide({vehicle}, {displacement}); immediate cost: 1
...
Candidate numbers may change between turns; use only the numbers above.
[/Environment update]
```

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of objects, each with "vehicle" as an uppercase vehicle label and "displacement" as a nonzero integer) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

无工具、无中途反馈，不要求逐步棋盘；完成、最优成本和格式分别计分。

#### 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Sliding vehicle {vehicle} by {displacement} is legal and preserves a minimum-move route to the goal.
<action>{local_id}</action>
```

排序包含最近至多10个候选及`current`，用`<`与`=`；须由实际冻结Q生成，不用精确标签冒充地图。Reason仅在裁判验证后使用；尚未生成SFT数据。无地图组删除排序行。

直接完成：

```text
Reason: This slide reaches the required goal.
<action>{local_id}</action>
```

已经完成：

```text
Reason: The required goal is already reached; no further move is needed.
<done/>
```

#### 完整计划结构说明例

仅说明格式，不进入测试题提示。

```json
{"actions": [{"vehicle": "A", "displacement": 4}], "final_status": "solved"}
```


### 数据说明例

```json
{
  "initial": "............AA......................",
  "goal_board": "................AA..................",
  "actions": [
    {
      "vehicle": "A",
      "displacement": 4
    }
  ],
  "reference_cost": 1,
  "role": "format illustration only"
}
```


## Token Swapping（图上交换棋子）

来源：`tasks/token_swapping/prompts/README.md`

### Token Swapping：任务文字与监督格式

采用[顺序Token Swapping](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2016.66)，公开实例来源为[tket固定问题](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/Data/FixedCompleteSolutions.hpp)。每点恰有一个不同棋子，目标为题目指定的完整排列；归位是其中一种目标。一次只交换一条边的两端，不使用并行轮数、部分映射或颜色等价规则。完整计划与指定目标裁判已接入；Qmap已完成多组训练但未通过验收，逐轮LLM训练尚未开展。历史Flash筛查的归位提示冻结在原suite中。

#### 共享任务描述与规则

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

#### 地图组／CoT 组：每轮环境更新

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

#### SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of arrays; each action contains the two distinct integer vertex IDs of an existing edge) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

完成、最优交换数与格式分别记录；多余交换计入成本，不能截去首次到达目标后的动作来获得最优分。

#### 监督答案：排序／Reason／Action

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

#### 完整计划结构说明例

仅说明格式，不进入测试题提示。

```json
{"actions": [[0, 1]], "final_status": "solved"}
```


### 数据说明例

```json
{
  "edges": [
    [
      0,
      1
    ],
    [
      1,
      2
    ]
  ],
  "initial": [
    1,
    0,
    2
  ],
  "goal": [
    0,
    1,
    2
  ],
  "actions": [
    [
      0,
      1
    ]
  ],
  "reference_cost": 1,
  "role": "format illustration only"
}
```
