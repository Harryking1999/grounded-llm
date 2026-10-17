# 五任务纯文字完整规划基线

**结果只看[模型、设置与预算对照表](results/report.md)。**初始五任务、旧积木复测和两组 5.6 逐步状态实验均已完成。新增 6.1-sol high 的 S/Z 同题对照运行中，定时跟进只监测该组。所有加难探索已停止，不再派题。

## 实验合同与答卷来源

所有条件一次输出完整计划，无工具、无中途反馈。裁判独立回放真实动作，模型自报棋盘不覆盖环境状态。每题一次有效答案，服务错误按配置有限重试；正常错误、截断和超限不重抽。历史 medium 每题八次，单列比较。

| 条件 | 正式配置 | 原始答卷目录（Git 外） |
|---|---|---|
| 6.1-sol high，初始五任务各 40 题 | [sol_high_40.json](configs/sol_high_40.json) | `runs/five_task_sol_high_40/` |
| Qwen3-32B thinking，同题 200 题 | [qwen32b_same40.json](configs/qwen32b_same40.json) | `runs/five_task_qwen32b_same40/` |
| 5.6-sol high，同题 200 题 | [sol56_high_same40.json](configs/sol56_high_same40.json) | `runs/five_task_sol56_high_same40/` |
| 6.1-sol high，旧十形状十二块 16 题 | [sol61_legacy_blocks12_new_prompt.json](configs/sol61_legacy_blocks12_new_prompt.json) | `runs/sol61_legacy_blocks12_new_prompt/` |
| 5.6-sol high，原八形状＋逐步状态 40 题 | [sol56_blocks40_state_report.json](configs/sol56_blocks40_state_report.json) | `runs/sol56_blocks40_state_report/` |
| 5.6-sol high，S/Z 八形状＋逐步状态 40 题 | [sol56_blocks40_sz_state_report.json](configs/sol56_blocks40_sz_state_report.json) | `runs/sol56_blocks40_sz_state_report/` |
| 6.1-sol high，逐字复用上述 S/Z 40 题 | [sol61_blocks40_sz_state_report.json](configs/sol61_blocks40_sz_state_report.json) | `runs/sol61_blocks40_sz_state_report/` |

提示词从 `tasks/<task>/prompts/README.md` 读取，生成后冻结在各运行目录的 `suite.json`。逐字同题组复制冻结内容；旧十形状保留原棋盘与形状目录，采用已确认的规则与完整计划 JSON；新 S/Z 八形状重新生成棋盘，与旧组不配对。

6.1 的 S/Z 组直接复制已冻结的 5.6 S/Z 题集，包括逐步状态要求，不再次生成棋盘；只有请求模型不同，高强度、16k 请求及服务重试设置保持一致。

## 评分与实现

主评分为 [16k 合同](configs/budget_16k.json)，附加 [8k 合同](configs/budget_8k.json)，都包含思考与最终答案；超限即错，用量缺失不计预算内成功。原执行分保留。请求上限阶段、第三方接口强度缺失与其他比较限制统一写在结果表中。

- `src/prepare.py`、`src/prepare_retest.py`：准备冻结题集并回放参考解。
- `src/run.py`：API 调用、断点续跑和有限服务重试；凭据只从进程环境传入。
- `src/qwen.py`：完整 BF16 Qwen3-32B、TP4、thinking，依赖见 [requirements-qwen.txt](requirements-qwen.txt)。
- `src/protocol.py`：独立环境裁判、合法动作和逐步状态判分。
- `src/budget.py`、`src/report.py`：预算重算与紧凑汇总。

针对性正确性检查：

```powershell
python -m unittest discover -s experiments/task_text_baselines/tests
```

## 按需重建明细

以下命令只读已有答卷，不调用模型。自动生成的 Markdown／JSON 放在忽略目录，`results/` 只维护统一结果表。

```powershell
python -m experiments.task_text_baselines.src.snapshot --baseline runs/five_task_sol_high_40 --qwen runs/five_task_qwen32b_same40 --out runs/task_text_reports
python -m experiments.task_text_baselines.src.retest_report --baseline runs/five_task_sol_high_40 --sol56 runs/five_task_sol56_high_same40 --legacy runs/sol61_legacy_blocks12_new_prompt --out runs/task_text_reports
python -m experiments.task_text_baselines.src.state_report --baseline runs/five_task_sol56_high_same40 --run runs/sol56_blocks40_state_report --out runs/task_text_reports
python -m experiments.task_text_baselines.src.state_report --baseline runs/sol56_blocks40_state_report --run runs/sol56_blocks40_sz_state_report --out runs/task_text_reports
python -m experiments.task_text_baselines.src.budget_report --qwen runs/five_task_qwen32b_same40 --sol56 runs/five_task_sol56_high_same40 --sol61 runs/five_task_sol_high_40 --legacy runs/sol61_legacy_blocks12_new_prompt --states runs/sol56_blocks40_state_report --sz runs/sol56_blocks40_sz_state_report --out runs/task_text_reports
```

已停止的加难原始答卷在 `runs/sol_difficulty_20/`；需要复核时给 `src.snapshot` 加 `--supplementary runs/sol_difficulty_20`，不把它们并入初始任务成绩。当前研究安排只维护[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。
