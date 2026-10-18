# 数字华容道 Qmap 运行记录

本轮按用户授权首次训练：覆盖全部复原可达状态，保留中间状态关系监督，最终以复原为主。配置见 [puzzle8.json](../configs/puzzle8.json)，方法与评测边界见 [README](../README.md)。

本地检查已通过：完整状态图、标准终点全状态距离与独立规则 BFS 一致；中间目标抽样距离核对；父节点和全部候选的对称关系无训练泄漏；真实梯度更新、断点读回和合法动作执行。40048 远端关键检查也通过。当前没有最终训练结果。

## 当前运行

- 源码：`6155519`；留在用户指定的当前分支 `codex/blocks-b2-data-diagnosis`，未推送。
- 机器：用户指定的 `root@172.16.78.10:40048`，GPU 1；项目 Python：`/zhanghanyue/experiment/flamingo_map_reader/.venv/bin/python`。
- 代码：`/zhanghanyue/experiment/grounded_llm_puzzle_qmaps/code/qmap_6155519`。
- 运行：`/zhanghanyue/experiment/grounded_llm_puzzle_qmaps/runs/puzzle8_6155519`；日志：同根目录 `logs/puzzle8_6155519.log`。
- 已完成精确数据构造并进入梯度更新；初始日志确认训练距离关系 4,536,000 条、候选决策 4,535,975 条，全状态复原距离覆盖已核对。目标与留出 cohort 由运行目录中的 `data_summary.json` 固定。
- 后台训练器自动选择 `best.pt`、保留 `last.pt`，随后写出复原及中间目标完整评测 `summary.json`。尚无最终精度或复原成功率结论。

GPU 0 的汉诺塔独立运行，积木诊断实验保持原样。依赖已在项目环境中补装 SciPy；实际版本随结果记录，不改动公共环境。
