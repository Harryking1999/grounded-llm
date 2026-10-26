# 24点：任务文字与监督格式

参考 [XoT的两数合并规则](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_all_in_one/xot/prompter/prompts/prompts_game24.py)及[公开环境](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/d69d3b8ddb75f888732e893394e8e5e5df4f4e6f/xot_mcts/game24/Game24Game.py)。保留题目及运算规则；完整计划采用本项目统一JSON，不引入上游few-shot、MCTS或多次求解。

以下定义正式文字协议。完整计划裁判与逐轮SFT适配分别维护；地图采用范围见同任务Qmap/selection.json，训练进展不在协议中维护。

## 共享任务描述与规则

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

## 完整多重集目标规则

```text
Reduce the initial sorted list to the complete goal multiset using exact rational arithmetic.
At each step choose two distinct current slots i<j, remove their numbers x,y and insert the result.
Sort the remaining list numerically after every operation. Slot indices start at0 and change after sorting.
Actions are[i,op,j]. Allowed ops:+ means x+y;- means x-y;r- means y-x;
* means x*y;/ means x/y;r/ means y/x. Division by zero is illegal.
Negative numbers, zero and fractions are allowed. Equal numbers occupy distinct slots.
Do not concatenate, introduce constants, split numbers or discard unused numbers.
Stop only when the entire sorted list exactly matches the goal, including repeated values.
Initial sorted list:
{initial_numbers}
Complete goal sorted list:
{goal_numbers}
```

## 地图组／CoT 组：每轮环境更新

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

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of exactly three arrays [i, op, j], using the current sorted slot indices and the allowed operation strings) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

沿用既有完整计划条件：无工具和中途反馈，不要求逐步状态报告。完成、最优成本与格式分别记录。模型和token预算以实验配置为准。

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Combining the numbers in slots {i} and {j} with {operation} is legal and leaves a solvable list with one fewer operation needed.
<action>{local_id}</action>
```

排序包含最近的至多10个候选及`current`，用`<`、`=`表示远近和并列。排序由实际冻结Q生成，上述Reason仅在裁判已验证相应事实时作为监督模板；数据须通过对应实验的认证。无地图CoT组只删除排序行，保留相同Reason和控制标签。

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

## 完整计划结构说明例

下例对应本任务[data/example.json](../data/example.json)，只说明格式。

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
