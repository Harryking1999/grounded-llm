# 手电过桥：任务文字与监督格式

规则采用 [Backhouse 与 Truong 的手电过桥问题](https://people.cs.nott.ac.uk/psarb2/papers/abstract.html)的容量2情形。当前六人时间组合是本项目诊断场景。

以下定义正式文字协议。完整计划裁判与逐轮SFT适配分别维护；地图采用范围见同任务Qmap/selection.json，训练进展不在协议中维护。

## 共享任务描述与规则

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

## 完整目标规则

```text
Find a minimum-total-time sequence of crossings from the initial configuration to the complete goal.
There are {people_count} people, numbered from0, two banks and one torch.
One or two distinct people cross with the torch. All travellers must be on the torch bank.
The travellers and torch move to the other bank. Cost is the maximum traveller time.
The same rules apply in either direction. Match EVERY person's bank and the torch bank exactly.
Use at most {action_limit} actions. A local candidate is the listed group of person IDs.
Person ID: crossing time
{crossing_times}
Initial configuration:
{initial_configuration}
Complete goal configuration:
{goal_configuration}
```

## 地图组／CoT 组：每轮环境更新

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

## SOTA 组／32B 组：输出格式

```text
Plan the complete remaining sequence from the current state and output it once. There is no intermediate feedback; the judge stops at the first illegal action.
Return JSON with "actions" (an array of arrays; each action contains one or two distinct integer person IDs) and "final_status" ("solved", "no_solution", or "unsolved").
Use "solved" when your sequence reaches the goal; if already at the goal, output an empty actions array. Use "no_solution" only when the goal is unreachable, not merely because you could not find a plan; otherwise use "unsolved".
```

沿用既有完整计划条件：无工具和中途反馈，不要求逐步状态报告。完成、最优成本与格式分别记录。模型和token预算以实验配置为准。

## 监督答案：排序／Reason／Action

```text
Map-distance ranking to the goal, closest to farthest: {ranking}.
Reason: Crossing with people {person_ids} is legal and preserves a minimum-total-time route to the goal.
<action>{local_id}</action>
```

排序包含最近的至多10个候选及`current`，用`<`、`=`表示远近和并列。排序由实际冻结Q生成，上述Reason仅在裁判已验证相应事实时作为监督模板；数据须通过对应实验的认证。无地图CoT组只删除排序行，保留相同Reason和控制标签。

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

## 完整计划结构说明例

下例对应本任务[data/example.json](../data/example.json)，只说明格式。

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
