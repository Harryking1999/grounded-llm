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
