# 推箱子

独立规则、数据适配和裁判已实现；首轮仅检验固定目标 Qmap 可行性。正式范围与预算见[配置](../../experiments/new_tasks_qmap/configs/sokoban.json)，结果见[报告](../../experiments/new_tasks_qmap/results/report.md)。不作为多任务SFT已采用任务。纯文字Flash64k固定筛查已完成，来源、结果及后续加难见[基线报告](../../experiments/task_text_baselines/results/report.md)。

- [规则与输出](prompts/README.md)
- [数据](data/README.md)与[示例](data/example.json)
- [地图](Qmap/README.md)与[采用记录](Qmap/selection.json)
- [环境及裁判](../../grounded_llm/tasks/sokoban/environment.py)

## 来源与适配

规则参考 [gym-sokoban](https://github.com/mpSchrader/gym-sokoban/blob/8e06e44e8bf3bb8bc73eeb1e7f0354508ce3fc89/gym_sokoban/envs/sokoban_env.py)；标准题集参照 [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels/blob/78011d8c86ad423bfb2e1005012393b4c3c24409/README.md)。后者为10×10、四箱。首轮是自生成的小棋盘诊断，不是 Boxoban 成绩；只采用四向推/走动作，不继承上游冗余 move-only、noop、奖励塑形或超时判胜。
