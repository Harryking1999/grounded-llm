# Rush Hour（移车出库）

采用[Fogleman公开题库与实现](https://github.com/fogleman/rush/tree/3e3b8396891d1802b30cb3129ae31fdedb319480)的6×6规则。车辆长2或3，只能沿自身轴线移动；一次滑动任意合法距离计1步，目标车A在棋盘内到达最右端即完成。完整计划文字与裁判已接入，尚无Qmap或LLM训练。

- [规则与文字](prompts/README.md)
- [题目来源与精确标签](data/README.md)
- [地图条件](Qmap/README.md)、[采用记录](Qmap/selection.json)
- [规则实现](../../grounded_llm/tasks/rush_hour/environment.py)
- [Flash筛查合同](../../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)、[统一结果](../../experiments/task_text_baselines/results/report.md)
