# 五任务纯文字完整规划基线

当前采用最初的五任务配置，每任务 40 题，同题比较 Qwen3-32B、GPT-5.6-sol high 与 GPT-6.1-sol high。三组均已完成，16k 预算内成功分别为 **38/200、175/200、183/200**。用户将最低难度要求改为 32B 不能可靠完成；GPT 作为强基线参照。最优解率另计。

## 输出预算

评分合同见 [budget_16k.json](configs/budget_16k.json)：16,000 输出 token，包含思考与最终答案；超过上限即计失败，用量缺失不计预算内成功。原始答卷与执行判分保留，报告派生预算分，不重新采样正常答卷。

接口曾在请求上限之外继续输出，因此请求参数不等于实际计算约束。历史答卷来自原请求上限，按 16k 重算不能冒充在 16k 下重新生成。5.6 已在旧请求收尾后，用现有 worker 加 `--budget-config experiments/task_text_baselines/configs/budget_16k.json` 完成剩余题；新答卷记录实际请求上限，progress 保留前后请求历史。即使接口超发，超限答案也不计成功。

当前紧凑结果见 [report.md](results/report.md)，对应数字见 [summary.json](results/summary.json)。运行中结果标注更新时间；尚未返回题与服务失败不当作模型答错。

正式参数见 [sol_high_40.json](configs/sol_high_40.json)。文字从 `tasks/<task>/prompts/README.md` 读取；生成后冻结题目、提示词与参考解在 Git 外的 `runs/five_task_sol_high_40/suite.json`。魔方共用固定动作教学示例，8-Puzzle 按用户要求无动作示例。各题只提交一次完整计划，无中途反馈。此条件与逐轮地图／CoT 反馈条件不同。

任务实现分文件，裁判不提供给被测模型。积木复用现有移除规则，魔方复用当前六动作角块环境；汉诺塔使用独立的八盘单目标 BFS，避免扩建旧六盘全状态对距离表；8-Puzzle 使用独立空格交换与 BFS。寻路复用已有固定三度无向图生成器。

主指标为真实执行到达目标的题数／40；JSON 与状态声明正确性、首次非法动作、合法但未完成、输出截断、服务错误、最优解率与 token 用量分别报告。基础设施失败可按配置有限重试；模型错误、截断不重试。原始回答和 API 元信息保留在运行目录，正式报告只提交紧凑摘要。

```powershell
python -m experiments.task_text_baselines.src.prepare --config experiments/task_text_baselines/configs/sol_high_40.json --out runs/five_task_sol_high_40
python -m unittest discover -s experiments/task_text_baselines/tests
# GND_API_KEY 仅通过进程环境传入；不写配置或命令行参数。
python -m experiments.task_text_baselines.src.run --out runs/five_task_sol_high_40
# 在原进程结束后恢复：保留正常答卷，优先补服务错误。
python -m experiments.task_text_baselines.src.run --out runs/five_task_sol_high_40 --recovery-config experiments/task_text_baselines/configs/service_recovery.json
python -m experiments.task_text_baselines.src.report --run runs/five_task_sol_high_40 --out runs/five_task_sol_high_40/summary
```

用户要求补齐 API 失败后，采用 [服务补测策略](configs/service_recovery.json)；只调整基础设施重试，不改变题目、模型、提示词、输出预算或正常答卷。观察到“服务更新中”的 HTTP 503 连续快速耗尽重试后，补测增加等待间隔并扩大有限次数；恢复时只重新激活仍有重试额度的服务失败任务。`stop_requested` 只停止新调度，等待在途请求落盘后才恢复；不可同时启动两个写同一运行目录的 worker。汇总将有效答卷与仍待补测的服务错误分开。

第三方接口的回传格式不一致：部分答卷明确回显模型名称、`reasoning.effort=high` 与 `mode=standard`，多数省略这些字段。报告分别统计请求值、实际回显及字段缺失，不把所有请求 high 写成所有答卷均已验证 high，也不把思考 token 数量当作档位证明。流式前序与完成事件的独立小探针同样未回传字段，因此并非仅因采集时覆盖了前序元信息。探针不计入评测；接口回显仍属于第三方声明，后台身份与计算配置不能独立验证。运行还记录服务端工具元信息与是否超出请求 token 上限。

## 同题 Qwen3-32B

用户指定在 028 使用 GPU 0–3。只读复用已有完整 BF16 基座，不下载量化版，不复制到 4090。正式推理合同见 [qwen32b_same40.json](configs/qwen32b_same40.json)：开启 Qwen thinking，输出预算包含思考与答案；该模式不等同于 GPT 的 high。依赖装在对应运行目录的独立虚拟环境，不修改共享项目环境。

依赖入口为 [requirements-qwen.txt](requirements-qwen.txt)。028 的原环境继承 `/ssdwork/miniconda3` 的 CUDA 库；隔离安装 torch 后需将继承环境和新环境中 `site-packages/nvidia/*/lib` 加到该进程的 `LD_LIBRARY_PATH`。缺失的 cuSPARSELt 按上述要求安装到新环境，不能修改共享 Conda。

`src/qwen.py` 读取同一个冻结 `suite.json`，保持全部题目与 user prompt 不变；任务轮流排队，每题只取一个样本，已完成题不重采样。截断和模型答错均保留。裁判只在生成后执行，不向模型提供搜索工具。

首批使用四题批量；观察到每批约五分钟、四卡缓存仍有足够容量后，后续改为十六题批量。采样参数、逐题种子与输出预算不变；原答卷保留，运行进度保存前后配置和源码来源，新答卷记录实际批量。不同批量可能引入底层数值差异，整轮仍只按每题首个有效答案计分。

在已提交源码快照目录执行：

```bash
python -u -m experiments.task_text_baselines.src.qwen --suite /path/to/frozen/suite.json --config experiments/task_text_baselines/configs/qwen32b_same40.json --out /path/to/qwen-run --model-path /ssdwork/fuzhizhang/model_base/Qwen3-32B --gpus 0,1,2,3 --source-commit COMMIT
```

原始逐题输出、进度和运行记录放在 `runs/five_task_qwen32b_same40/`，与 API 答卷分开；两者均可用 `src/report.py` 生成紧凑汇总。

028 恢复连接后已回收完整 200 题，真实执行解出 40 题；逐题提示词与 GPT 原题集完全一致，全部未截断答案重新回放后与原判分一致。完整对照及失败类型见 [结果报告](results/report.md)。无需重启已完成的评测。

## 新增模型与旧题复测

用户授权新增两组，每题一次有效答案，结果单列在[复测报告](results/model_retest.md)：

| 组别 | 题目与文字 | 配置 |
|---|---|---|
| 5.6-sol 五任务 | 逐字复用原五任务 200 题、提示词与参考解，不重新采样 | [同题合同](configs/sol56_high_same40.json) |
| 6.1-sol 旧十二块 | 原 16 张棋盘与含 S/Z 十形状；当前规则和完整计划 JSON，不要求 board_after | [旧题合同](configs/sol61_legacy_blocks12_new_prompt.json) |

两组已全部完成并补齐服务失败。5.6 五任务原执行成功 182/200，16k 预算内为 175/200；6.1 旧十二块为 16/16，包含历史 0/8 的三道题，全部未超 16k。5.6 的完整 200 份执行结果已重新回放，判分一致。定时跟进在本轮完成后停止。

两组均请求 high。原答卷保留，当前统一按 16k 评分；5.6 包含前后两个请求上限阶段，不能冒充严格等生成预算。旧十二块的构造解只保证可解，不作为精确最短解。历史 5.6 使用 medium、旧提示词、每题八次，不能用该历史差值单独判断模型升级或逐步状态输出的贡献。要单独验证逐步状态输出的影响，应固定模型、题目与预算，仅改变是否要求 board_after；该对照尚未运行。

```powershell
python -m experiments.task_text_baselines.src.prepare_retest --config experiments/task_text_baselines/configs/sol56_high_same40.json --out runs/five_task_sol56_high_same40
python -m experiments.task_text_baselines.src.prepare_retest --config experiments/task_text_baselines/configs/sol61_legacy_blocks12_new_prompt.json --out runs/sol61_legacy_blocks12_new_prompt
# 为两个目录分别调用现有 src.run；仅从环境传入 GND_API_KEY。
python -m experiments.task_text_baselines.src.retest_report --baseline runs/five_task_sol_high_40 --sol56 runs/five_task_sol56_high_same40 --legacy runs/sol61_legacy_blocks12_new_prompt --out experiments/task_text_baselines/results
```

复测队列只重试服务错误，正常答错和截断均保留；原始回答位于上面的忽略目录。后台将最新汇总写入 `runs/model_retests/summary/`；受跟踪报告是注明更新时间的回收快照，避免后台持续修改源码工作区。

## 当前范围与补充探索

用户追加授权 [积木逐步状态对照](configs/sol56_blocks40_state_report.json)：5.6-sol high 重做原 40 题，仅在冻结提示词后追加每步 `board_after`（十行、每行十位的完整棋盘）要求，仍一次输出完整计划。新组请求与评分上限均为 16k；与原无状态组 37/40 配对比较执行成功、预算内成功、严格合同和合法前缀逐步状态准确率。模型自报状态不反馈给环境。原组混用请求上限的限制仍明确注明。

```powershell
python -m experiments.task_text_baselines.src.prepare_retest --config experiments/task_text_baselines/configs/sol56_blocks40_state_report.json --out runs/sol56_blocks40_state_report
python -u -m experiments.task_text_baselines.src.run --out runs/sol56_blocks40_state_report --budget-config experiments/task_text_baselines/configs/budget_16k.json
python -m experiments.task_text_baselines.src.state_report --baseline runs/five_task_sol56_high_same40 --run runs/sol56_blocks40_state_report --out experiments/task_text_baselines/results
```

原始答卷在上述忽略目录；后台汇总写入 `runs/sol56_blocks40_state_report/summary/`，正式报告为回收快照。

题集配置仍为 [sol_high_40.json](configs/sol_high_40.json) 与 [qwen32b_same40.json](configs/qwen32b_same40.json)，评分另由上述预算合同决定。两组各 200 题均已完成；原执行成功 32B 40、GPT 195，按 16k 重算为 32B 38、GPT 183，完整设置见[主报告](results/report.md)。题目未因筛选或评分标准改变而重新采样。

所有加难配置现已停用，包括 `difficulty_20.json`。运行目录中的 `stop_requested` 阻止继续派题，已发出的请求正常落盘后退出。15×15 积木、三阶魔方、5×5 华容道和十一盘汉诺塔只作为[补充探索](results/difficulty_exploration.md)保留，不要求测满其原计划，也不与原各 40 题合并。更早的五题初筛、28 块积木和寻路加难同样保留原始答卷，不再运行。

停止加难不改变任务自身的规则，也不自动触发新训练。后续地图／无地图／7B 的配对比较沿用确定的主任务条件；现有权重若尺寸或动作制不适用，应明示并另行准备，不以旧权重冒充已完成的新条件。

## 重建紧凑报告

`src/snapshot.py` 只读取保存的答卷，不发 API 请求。主报告始终以最初配置和同题 32B 对照为中心；加难结果单独生成补充材料，避免旧队列状态再次覆盖当前主线。

```powershell
python -m experiments.task_text_baselines.src.snapshot --baseline runs/five_task_sol_high_40 --qwen runs/five_task_qwen32b_same40 --supplementary runs/sol_difficulty_20 --out experiments/task_text_baselines/results
```

可选 `--history` 将更早停用轮次的摘要保存在机器可读结果中；原始题目、回答、服务失败、复用来源与配置快照继续保存在 Git 外的 `runs/`。
