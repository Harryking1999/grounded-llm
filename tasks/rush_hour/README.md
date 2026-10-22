# Rush Hour（移车出库）

采用[Fogleman公开题库与实现](https://github.com/fogleman/rush/tree/3e3b8396891d1802b30cb3129ae31fdedb319480)的6×6规则。车辆长2或3，只能沿自身轴线移动；一次滑动任意合法距离计1步，目标车A在棋盘内到达最右端即完成。完整计划文字与裁判已接入，尚无Qmap或LLM训练。

第一版采用同一个Qmap覆盖多种车辆布局，编码完整棋盘，按整个布局／可达分量留出测试。每题指定一张A到右端、其他车辆位置也明确的完整成功目标棋盘，训练与评测均要求精确匹配；这属于原出口目标的变体，不能用旧Flash成绩替代。构建与验收边界见[Qmap](Qmap/README.md)。

Flash high、64k筛查完成17/40，全部成功计划最优；21题思考耗尽，2题对可解题误报无解。40起点来自8个公开无墙分量，最短2至51步，不能当作40道独立原题的总体成绩。证据与限制见下方统一结果。

- [规则与文字](prompts/README.md)
- [题目来源与精确标签](data/README.md)
- [地图条件](Qmap/README.md)、[采用记录](Qmap/selection.json)
- [规则实现](../../grounded_llm/tasks/rush_hour/environment.py)
- [Flash筛查合同](../../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)、[统一结果](../../experiments/task_text_baselines/results/report.md)
