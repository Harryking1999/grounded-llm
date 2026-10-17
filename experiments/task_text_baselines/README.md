# 五任务纯文字完整规划基线

用户授权为积木、寻路、汉诺塔、二阶魔方与 8-Puzzle 各构造 40 题，测试 GPT-6.1-sol high。检验既定文字外壳下的完整计划执行成功率，按任务与难度分层观察失败；不使用地图或模型侧工具。

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

第三方接口短探针接受了请求并正常返回，但省略返回模型名称与 reasoning 配置。报告分别记录请求值与返回值，不把请求标签视为独立身份验证。运行还记录服务端工具元信息与是否超出请求 token 上限。

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

## 当前加难队列：每档至少 20 题

唯一在用配置：[difficulty_20.json](configs/difficulty_20.json)。初筛和独立确认均为 20 道不同题；有效答卷不足时只报进度，不用旧五题样本宣布完成。

| 任务 | 当前条件与边界 |
|---|---|
| 积木 | 用户指定 15×15、20 次非重叠放置、含 S/Z 的十种固定形状；撤下 28 块条件 |
| 三阶魔方 | 扩充当前 20 步打乱档至 20 题；仍偏易则按配置继续，始终保持三阶 |
| 华容道 | 只尝试 5×5；保证可解，Manhattan 下界至少 60，不提供动作示例；若仍偏易就停止该方向，不升至 6×6 |
| 汉诺塔 | 当前 11 盘档扩充至 20 题，保留先前同题答案 |
| 寻路 | 用户明确停止加难，不进入新队列 |

各组一次输出完整计划，无中途反馈，客户端不提供工具；规则和输出文字沿用任务原文，仅替换尺寸、数字范围与形状表。积木 20 指构造放置次数，不强制模型找回原来 20 块；任意合法分解都接受。构造／打乱长度仅是参考解上界，华容道 Manhattan 值是下界，均不冒充精确最短距离。汉诺塔继续使用精确距离。

先停止旧队列派新题并等待在途请求落盘，再冻结四组各 20 题及回放通过的参考解。`src/prepare_difficulty.py` 仅复用状态、目标、提示词完全相同且可复现判分的有效答卷，包括答错的答卷；原始服务失败记录一并保留来源。新旧运行分开，旧五题初筛作为历史探索保留，不替代新的分母。

```powershell
python -m experiments.task_text_baselines.src.prepare_difficulty --config experiments/task_text_baselines/configs/difficulty_20.json --out runs/sol_difficulty_20
python -u -m experiments.task_text_baselines.src.escalate --config experiments/task_text_baselines/configs/difficulty_20.json --out runs/sol_difficulty_20
```

每档测完才判断是否继续；成功不高于一半时，用同条件 20 道独立新题确认。计划执行错误、未完成、格式问题、输出截断与服务失败分别报告；成功但格式不合规仍计解出，服务失败不计答错，正常答案不重采样。接口已观察到实际输出超过请求 token 上限，须单列，不能声称严格等预算。

原始 200 题及其 Qwen3-32B 对照保持不变。`difficulty_ladder.json`、两组 4×4 距离压力配置、密集积木及长路径配置已标记停用，旧答卷保留在对应 `runs/`；队列入口拒绝再次用停用配置启动。新运行与报告位于 `runs/sol_difficulty_20/`，项目进度仍只更新权威状态页。
