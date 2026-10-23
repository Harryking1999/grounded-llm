# 移车出库数据

来源为[Fogleman公开题库](https://www.michaelfogleman.com/rush/)及其[题目预览文件](https://www.michaelfogleman.com/static/rush/rush1000.txt)。本项目只使用无墙6×6布局，并独立核对可达状态和最短距离。

## 当前数据

共有224个场景：144个训练、35个验证、37个测试，另有8个已知题；总计1,348,182个状态。一个场景包含固定车辆配置及其可达局面，同一配置下的场景不会跨训练与测试组。具体范围见[数据统计](../../../experiments/new_tasks_qmap/results/shared/exact_pairs_data.json)和[配置](../../../experiments/new_tasks_qmap/configs/rush_hour/rush_hour_pairs.json)。

每题指定一张完整目标棋盘，所有车辆位置明确，A在最右端。距离重新计算到这个目标，一辆车滑动任意合法距离均计一步。

## 如何理解来源成绩

上游的最短步数只针对A到出口，不能直接当作本项目完整目标的距离。同一原题产生的中间起点也不算独立新布局。

早期DeepSeek筛查使用8个场景中的40个起点；它与当前完整目标评测分开记录，见[文字报告](../../../experiments/task_text_baselines/results/report.md)。下载的原始文件和实际选题保存在运行数据中。[example.json](example.json)仅说明格式。
