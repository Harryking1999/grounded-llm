# Grounded LLM with Cognitive Maps

研究显式状态模型能否帮助 LLM 更准确地估计状态、预测动作后果并完成规划。当前只研究寻路与积木；共享状态表示、跨任务迁移与交互学习属于更远期问题，尚未得到实验验证。

当前主线是把学到的 cognitive map／roadmap 作为显式外部 world model：它维护状态，提供候选动作后果及与目标的地图距离，由 LLM 读取这些信息并规划。寻路 Step 1 已验证 Q/V 地图的转移与远近关系；积木下一步探索从合法转移学习 Q/V。此前的连续 state token 对齐属于 previous approach。

## 阅读入口

- [当前状态与有序 TODO](docs/PROJECT_STATUS_AND_TODO.md)：唯一当前进度页。
- [研究简述](docs/RESEARCH_BRIEF.md)：研究定义、已确定约束与证伪标准。
- [Step 1 结果](experiments/cml_map_scaling/results/report.md)：地图规模、表示几何与长边诊断。
- [图 Step 2 旧方案](experiments/cml_map_scaling/STEP2_RUNBOOK.md)、[积木 Step 2 旧方案](experiments/state_interface_pilot/README.md)：连续 state token 路线，保留供复现。
- [Related Work](RELATED_WORK.md)：GCML、状态表征维护与 RAP，含摘要、首图和参考意义。
- [Qwen 验收报告](experiments/qwen_path_blocks/results/report.md)：最终双向 path256 与 blocks 16k，含准确率、截断重分类与案例。
- [Sol 基线](experiments/sol_dag_blocks/README.md)、[Luna / Flash 基线](experiments/gcml_counterexamples/README.md)：历史独立条件。
- [GCML 任务来源](docs/GCML_TASKS.md)。

## 仓库约定

实验在 `experiments/<study>/` 下保存说明、正式配置、可复用源码、针对性测试及紧凑结果。原始生成、日志、数据和模型放在忽略目录中。协作与 Git 纪律见 [AGENTS.md](AGENTS.md)。
