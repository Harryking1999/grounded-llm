# 纯文字完整规划基线

**结果只看[模型、设置与预算对照表](results/report.md)。**初始五任务、DeepSeek Flash 同题 200 题、旧积木复测、逐步状态、S/Z 八形状仅动作及完整十形状 100 题双模型对照均已完成。服务失败已补齐，定时跟进已暂停。此前加难探索均已停止，不再派题。

DeepSeek 积木／二阶魔方新题复测见 [64k 合同](configs/deepseek_blocks_cube100_64k.json)与[结果表](results/report.md)。原始答卷、进度与失败记录仅存 Git 外 `runs/deepseek_blocks_cube_fresh100_64k/`。

手电过桥、推箱子、24点的固定设置Flash筛查见[64k合同](configs/deepseek_new_tasks_64k.json)。112题已完成，三项全部成功；完成与最优成本分别计分，来源为既有Qmap留出题，推箱子是小棋盘诊断条件。后续按[首次错误合同](configs/deepseek_new_tasks_harder64k.json)使用现有`src.escalate`，每任务每档至多三题，出现首个有效未完成即停止。过桥增加人数与时间范围，推箱子改用固定提交的原Boxoban文件，24点先取XoT必须分数的四数题，再扩数值范围；参考轨迹先独立回放。只找失败位置，不据少量自适应题估计总体成功率。

## 实验合同与答卷来源

Rush Hour／Token Swapping两项筛查见[64k合同](configs/deepseek_rush_token64k.json)与[结果表](results/report.md)：各40题，共80题，分别完成17/40、36/40，完成答案均最优；25题思考耗尽，Rush Hour另2题误报无解。前者为8个公开无墙分量各5个距离起点，后者为tket公开固定问题的8种架构各5题。精确最优值重算、完成与成本分开评分，复用`src.run`。准备入口为`src.prepare_rush_token`，原始题目与响应在`runs/deepseek_rush_token64k/`；未训练地图。

所有条件一次输出完整计划，无工具、无中途反馈。裁判独立回放真实动作，模型自报棋盘不覆盖环境状态。每题一次有效答案，服务错误按配置有限重试；正常错误、截断和超限不重抽。历史 medium 每题八次，单列比较。

| 条件 | 正式配置 | 原始答卷目录（Git 外） |
|---|---|---|
| 6.1-sol high，初始五任务各 40 题 | [sol_high_40.json](configs/sol_high_40.json) | `runs/five_task_sol_high_40/` |
| Qwen3-32B thinking，同题 200 题 | [qwen32b_same40.json](configs/qwen32b_same40.json) | `runs/five_task_qwen32b_same40/` |
| 5.6-sol high，同题 200 题 | [sol56_high_same40.json](configs/sol56_high_same40.json) | `runs/five_task_sol56_high_same40/` |
| DeepSeek Flash high，初始五任务同题 200 题，16k | [deepseek_flash_same40_16k.json](configs/deepseek_flash_same40_16k.json) | `runs/deepseek_flash_same40_16k/` |
| DeepSeek Flash high，清空／非空目标各 50 新题，64k | [deepseek_blocks100_goal_mix_64k.json](configs/deepseek_blocks100_goal_mix_64k.json) | `runs/deepseek_blocks100_goal_mix_64k/` |
| 6.1-sol high，旧十形状十二块 16 题 | [sol61_legacy_blocks12_new_prompt.json](configs/sol61_legacy_blocks12_new_prompt.json) | `runs/sol61_legacy_blocks12_new_prompt/` |
| 5.6-sol high，原八形状＋逐步状态 40 题 | [sol56_blocks40_state_report.json](configs/sol56_blocks40_state_report.json) | `runs/sol56_blocks40_state_report/` |
| 5.6-sol high，S/Z 八形状＋逐步状态 40 题 | [sol56_blocks40_sz_state_report.json](configs/sol56_blocks40_sz_state_report.json) | `runs/sol56_blocks40_sz_state_report/` |
| 6.1-sol high，逐字复用上述 S/Z 40 题 | [sol61_blocks40_sz_state_report.json](configs/sol61_blocks40_sz_state_report.json) | `runs/sol61_blocks40_sz_state_report/` |
| 5.6-sol high，S/Z 同题，仅动作 | [sol56_blocks40_sz_actions.json](configs/sol56_blocks40_sz_actions.json) | `runs/sol56_blocks40_sz_actions/` |
| 6.1-sol high，S/Z 同题，仅动作 | [sol61_blocks40_sz_actions.json](configs/sol61_blocks40_sz_actions.json) | `runs/sol61_blocks40_sz_actions/` |
| 6.1-sol high，完整十形状，新 100 题，仅动作 | [sol61_blocks100_sz10_actions.json](configs/sol61_blocks100_sz10_actions.json) | `runs/sol61_blocks100_sz10_actions/` |
| 5.6-sol high，逐字复用上述 100 题，仅动作 | [sol56_blocks100_sz10_actions.json](configs/sol56_blocks100_sz10_actions.json) | `runs/sol56_blocks100_sz10_actions/` |

提示词从 `tasks/<task>/prompts/README.md` 读取，生成后冻结在各运行目录的 `suite.json`。逐字同题组复制冻结内容；旧十形状保留原棋盘与形状目录，采用已确认的规则与完整计划 JSON；新 S/Z 八形状重新生成棋盘，与旧组不配对。

DeepSeek 组逐字复制初始五任务 200 题，积木采用原八形状、12 次构造、仅动作输出。官方当前 Flash 接口对应 V4.1-Flash-1017，旧 V4 Flash 名称已转接，在线模型列表未提供 R1；不得将本轮记为原 V4 或 R1 实测。通过现有 Responses 入口请求 high，思考和答案共限 16k；返回身份、强度、用量和截断保留在原始答卷。进度及结果见统一结果表。

6.1 的 S/Z 组直接复制已冻结的 5.6 S/Z 题集，包括逐步状态要求，不再次生成棋盘；只有请求模型不同，高强度、16k 请求及服务重试设置保持一致。

S/Z 仅动作两组只删除冻结提示词末尾的逐步状态指令，保留全部题目、规则、参考解及动作／终态 JSON；5.6 和 6.1 逐字使用同一提示词。两组与各自已完成的逐步状态组构成同题输出要求对照。

完整十形状 100 题复用历史目录和相邻构造：每步从全部可行、不重叠且相邻的放置中均匀选择，排除平移等价重复及历史题。保留已确认的当前规则和动作／终态 JSON，不要求状态报告；两个模型逐字同题同提示词，旧 16 题单列，不混入新 100 题。

## 评分与实现

主评分为 [16k 合同](configs/budget_16k.json)，附加 [8k 合同](configs/budget_8k.json)，都包含思考与最终答案；超限即错，用量缺失不计预算内成功。原执行分保留。请求上限阶段、第三方接口强度缺失与其他比较限制统一写在结果表中。

- `src/prepare.py`、`src/prepare_retest.py`：准备冻结题集并回放参考解。
- `src/prepare_new_tasks.py`：从新三任务既定留出条件抽取筛查题并回放参考解；原始题集和答卷在 `runs/deepseek_new_tasks_64k/`。
- `src/blocks_goal_mix.py`：清空／非空目标混合题；用差集精确覆盖核验非空目标最短步数，孤立格按上下左右邻接判断。
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
