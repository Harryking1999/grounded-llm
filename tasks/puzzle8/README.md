# 8-Puzzle（数字华容道）任务工作区

独立任务：3×3 棋盘、八个数字块和一个空格。规则、文字基线与Qmap已实现；采用的复原权重及适用范围见[selection](Qmap/selection.json)，中间目标能力仍有限。纯文字基线的[环境与BFS](../../grounded_llm/tasks/puzzle8/rules.py)、[题集准备和评测](../../experiments/task_text_baselines/README.md)与地图条件分别解释。

- [任务规则、每轮更新与输出草案](prompts/README.md)
- [数据说明](data/README.md)与[结构示例](data/example.json)
- [Qmap 方法与采用范围](Qmap/README.md)
- 当前进展与顺序维护在[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

参考 [XoT 的 8-Puzzle 提示词](https://github.com/microsoft/Everything-of-Thoughts-XoT/blob/main/xot_all_in_one/xot/prompter/prompts/prompts_npuzzle.py)：数字矩阵、0 表示空格、动作方向表示空格移动方向。新任务独立维护环境、数据、训练与评测，不混入魔方或汉诺塔训练器；基础编码器与关系监督方法对齐。
