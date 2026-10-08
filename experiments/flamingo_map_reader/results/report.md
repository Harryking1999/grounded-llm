# 积木分离 K/V：重训结果

2026-10-08 按会议决议停止融合 FFN 运行，恢复原分离 K/V、无 FFN 接口，从头训练。当前尚无新训练的评测成绩。

普通物理任务保留旧编号增强；已达标与无解各保持原数量，不增强。正式设置见[设计](../DESIGN.md)、[训练合同](../configs/blocks_kv_restart.json)与[三任务评测合同](../configs/blocks_kv_evaluation.json)。新结果按三类任务独立报告，不跨任务加权。

启动准备及针对性检查已完成，等待远程启动检查回填。机器运行记录保存在忽略的 `runs/`。

历史参考：[融合 FFN 到 epoch 5 报告](report_ffn_failure.md)及[完整计数](blocks_ffn_failure_summary.json)；[原分离 K/V 两任务报告](report_long_trajectory.md)。融合 FFN 停机时约 5.36 epoch，正式参考冻结至已全量评齐的 epoch 5，额外断点仍保留。
