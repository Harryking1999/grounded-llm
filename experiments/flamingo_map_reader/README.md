# 地图读取：完整轨迹 SFT

本轮按[飞书 4.2](https://zcnhpsd26tyi.feishu.cn/wiki/ZNNSwpIo0iHwMgk7mk7coBA3nzb#BJmUde2p6oN7SxxELqkcIfUEnIe)组织两项实验：寻路只用一张地图及其 Q，积木用 1000 张初始棋盘与共享 Q 编码器。冻结 Qwen2.5-1.5B-Instruct 和地图，两任务分别训练接口。

**训练单位是完整轨迹。**每轮学习状态判断、距离数值、排序、动作；到达后还要学习终止与真实动作总结。同一物理轨迹使用多套自洽随机编号，每套重复多次。测试自由生成每一轮及完整闭环，不设前置距离读出实验。

- [训练与评测方案](DESIGN.md)：数据规模建议、编号覆盖、监督格式及评价口径。
- [项目状态与 TODO](../../docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页。
- [结果报告](results/readout_failure_analysis.md)：新实验暂无成绩，旧结果已标为历史。

## 实现基础与需要修改的位置

| 环节 | 现有实现 | 本轮需要完成 |
|---|---|---|
| 寻路采样 | `src/data.py`、`src/prepare.py` | 单图、宽长度、训练／验证／测试目标分区及终止样本 |
| 积木采样 | `src/blocks_data.py` | 1000 张训练棋盘、长短任务、足够大的验证／测试及状态覆盖标注 |
| 轨迹与标签 | `src/sft.py`、`src/blocks_sft.py` | 距离数值监督；固定物理轨迹后生成编号版本 |
| 数据重复 | `src/train.py` 的完整轨迹 Dataset | 多套固定编号，每套重复；旧固定 seed 不能满足 |
| 地图绑定 | `src/transcript.py`、`src/memory.py`、`src/fusion.py` | 保留逐轮绑定，验证完整动作标记与终止段边界 |
| 评测 | `src/evaluate_graph.py`、`src/evaluate_blocks.py`、`src/relations.py` | 新任务集、全部轮次、数值与总结评分、目标替换和无地图消融 |

现有代码能提供完整轨迹 SFT 的基础，但还不能直接运行新方案。新增距离输出后需检查真实长样本的长度和显存，不能截掉后续轮或终止段。正式配置在采样与 smoke 后填写实际规模和运行预算。

## 历史配置与结果

原[寻路 K/V 配置](configs/pilot_path256_addressed_kv.json)和[积木 K/V 配置](configs/pilot_blocks_addressed_kv.json)是架构及优化器的参考，**仍是旧实验合同**。不修改旧合同来冒充新运行，不直接执行旧四图／五图评测作为本轮结果。

[16／4 小样本与原完整轨迹结果](results/archive/full_trajectory_history.md)已标为历史；[swap 条件](results/archive/swap_training.md)继续归档。旧自然目标、二选一、排序加权等分支不进入本轮训练。

代码验证入口仍为：

```bash
python -m unittest discover -s experiments/flamingo_map_reader/tests -p 'test_*.py'
```

新数据、日志和权重放在 Git 外的 `runs/` 等目录。源代码、正式配置与紧凑结果分别提交；寻路与积木保存独立接口权重。
