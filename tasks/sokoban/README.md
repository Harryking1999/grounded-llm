# 推箱子

保留的新增条件为**Boxoban hard：10×10棋盘、4个箱子、4个目标、1名玩家**，使用公开原题的布局。hard是上游策略求解失败筛出的子集，不是更大棋盘或更多箱子。同一个Qmap读入完整棋盘、覆盖多张hard布局，按未见布局检验泛化；每题指定一张箱子已到位且玩家位置明确的完整目标棋盘，训练与评测均要求精确匹配。动作、文字和指定目标裁判可复用；该条件尚无正式Qmap训练数据、采用地图或LLM训练。

原题362曾使Flash high在64k预算内耗尽思考、最终答案为空，本地已有52次合法走／推动作的见证；52不是精确最短走动距离。它证明该实例在此预算下出现失败，不证明整个hard子集的模型成功率，也不表示任务本身无解。来源与结果见[文字基线报告](../../experiments/task_text_baselines/results/report.md)。

历史首轮Qmap为6×6两箱诊断，见[原配置](../../experiments/new_tasks_qmap/configs/sokoban.json)与[原报告](../../experiments/new_tasks_qmap/results/report.md)；不能用于声称已有10×10四箱地图。

- [规则与输出](prompts/README.md)
- [数据](data/README.md)与[示例](data/example.json)
- [地图](Qmap/README.md)与[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/sokoban/environment.py)

## 来源与适配

初次阅读可把它理解为“人在仓库里把箱子推到地上的指定位置”。墙体是不能走的格子，目标点是箱子的停放格；两者在一题内不移动，不同题可以不同。完整棋盘编码同时保留墙、目标、箱子和玩家，见[Qmap输入说明](Qmap/README.md#规则状态与输入)。

规则参考 [gym-sokoban](https://github.com/mpSchrader/gym-sokoban/blob/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89/gym_sokoban/envs/sokoban_env.py)；布局采用 [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels/blob/78011d8c86ad423bfb2e1005012393b4c3c24409/README.md)的hard子集。只采用四向推／走动作，每次走动或推动均成本1；箱子不能被拉动或同时推动多个。原题目标为全部箱子到位、玩家终点不限；首版Qmap增加指定玩家终点，属于明确标注的目标变体。合法但进入死路的动作仍是合法候选。完成、最短走／推动作数与推箱次数分开，不能把最少推动数当作最少走动数。不继承上游冗余move-only、noop、奖励塑形或超时判胜。
