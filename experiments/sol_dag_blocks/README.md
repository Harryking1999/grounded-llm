# Sol：新形状积木与单向无环路径初测

按照 2026-09-07 用户确认的实验修改方案，单独测试 8-block、12-block、32 节点 DAG，各 16 个实例、每题八次采样，共 384 次正式试验。只测 `gpt-5.6-sol`、`medium`。完整运行合同以 [`configs/sol_medium.json`](configs/sol_medium.json) 为准。当前已完成离线生成与裁判验证，API 结果待运行后填写。

## 假设与边界

本轮检验：明确禁止移除空格、移除双格形状并要求逐步自报棋盘后，直接语言模型是否仍出现动作非法、状态预测错误或未清空；在随机定向的 DAG 中能否遵守方向并找最短路径。模型一次输出完整计划，无中途反馈、无工具、无训练组件。精确参考路径与积木构造见证只供离线裁判使用。

这不是旧 Luna/Flash 的同题、同提示词对照，不能将成功率差异归因于模型或逐步状态报告。两组积木均重新生成；不再使用含双格的官方 8-block 样本。“8/12 块”指生成块数，既不限制答案步数，也不假定是最少移除数。本轮不计算精确最少移除数。

## 输入与规则

形状顺序与编号只在配置中定义，提示词由配置自动生成。保留四种 L 三格、横竖三格，去掉横竖双格；用户已确认新增 **S/Z 四格四种朝向（含镜像）**。旧 6/7 号横竖三格重编号为 4/5；新增形状为 6–9。分别按单个 shape_id 与 L3、I3、SZ4 子类统计。`Blocks8Task` 和 `Blocks12Task` 继承统一的 `BlocksTask`，避免规则分叉。

棋盘由不重叠且相邻的合法放置构成，每次从全部可行放置中均匀抽样。此规则不保证各形状等频；输入频率会实测报告。两组之间也排除平移等价棋盘。没有按陷阱、求解难度或模型表现筛选。允许任意合法分解、无限复用固定朝向的形状。形状的 1 格必须全是棋盘当前占用格；形状的 0 格是空洞，不移除也不约束所覆盖的棋盘格。越界、初始空格重叠与重复移除均非法。

DAG 使用原 32 节点无向图为骨架，每个实例独立抽隐藏拓扑顺序，将所有边沿该顺序定向，确保无环。依用户补充要求，只在最短距离至少 4 步的有序点对中均匀抽一个起终点；没有合格点对的图重新生成。独立打乱节点行及各出邻居列表，同题八次提示词固定。隐藏顺序与距离筛选阈值不发送模型，无开关或门，禁止回访，要求最短路。这是按距离筛选的结构挑战集，不代表随机可达点对的总体难度分布。

积木每步输出 `shape_id`、`row`、`col`、`board_after`（十条十位二进制字符串），最后输出 `final_status`。路径每步输出 `from/to`，最后输出 `final_node`。状态均为模型预测；执行裁判绝不相信报告的棋盘来更新真实状态。

## 指标

- 任务成功率：积木全程合法并清空；路径全程合法且最短。另报路径合法到达率。
- pass@8：每个固定实例八次试验至少一次任务成功；截断计失败，服务错误单独归档，禁止重采已完成的错误答案。
- 完整合同通过率：任务成功、严格 JSON、最终状态报告正确，且积木所有动作后的棋盘报告正确。
- 非法答案率、首个非法动作类型；仅在裁判可执行的合法前缀上评估棋盘报告，缺失或无效报告计错，另报有效报告率及逐格准确率。非法动作之后不继续假定执行。
- 输入 shape 频率指构造见证的形状计数（每题只计一次）；输出频率分别统计所有可解析动作、合法前缀和成功答案，按形状及子类报告计数和归一化频率。构造分解不唯一，频率差不能直接解释为某种形状识别概率；另按输入含某类形状的棋盘条件报告输出出现率。
- 原始 API 响应、实际 token 用量、请求/返回模型名及推理设置保留在 Git 忽略的 `runs/`。请求预算与网关实际执行值应区分。

## 运行与复现

从仓库根目录运行（仅依赖 Python 标准库）：

```powershell
python -m unittest discover -s experiments/sol_dag_blocks/tests -v
python experiments/sol_dag_blocks/src/prepare.py
# 在当前进程环境中设置 GND_API_KEY；不要将密钥写入文件或提交。
python experiments/sol_dag_blocks/src/run.py --api-config experiments/sol_dag_blocks/configs/sol_medium.json --out experiments/sol_dag_blocks/runs/sol_medium
python experiments/sol_dag_blocks/src/analyze.py --run experiments/sol_dag_blocks/runs/sol_medium/run.json
```

生成器拒绝覆盖已有 `runs/suite.json`。配置须先提交再正式运行。`run.py` 复用已有 `gcml_counterexamples/src/run.py` 的 Responses API 调用与续跑校验接口，使用本研究调度、提示词和裁判；旧实验源码不变。首个请求使用正式样本并计入 384 次，成功完成后以配置中的并发数推进；同合同续跑已有完成答案时直接恢复并发。因实际遇到网关并发限制和服务器过载，后续阶段仅对明确的限流/服务错误有限重试，重试次数与总服务故障上限见配置；失败槽位排在尚未尝试的槽位后。若明确触发网关并发限制，运行期并发逐次下降并记录实际值。单槽位耗尽重试后保持未完成并继续其他试验；达到全局服务故障上限时停止补充新请求，收齐在途结果后保存。已确认服务错误也可用 `--continue-from` 指向旧记录并选择新输出目录补齐，保留所有完成或预算截断试验。运行中可创建 `runs/stop_requested`，调度器停止补充新请求、收齐在途结果后退出；续跑前移除该标记。

用户在正式运行开始后要求从四并发提高到十二并发。初始四个完成答案全部保留；旧调度器不支持在线调整，切换时最多四个可能在途槽位单独归档为客户端中断，并在新阶段补齐。中断请求未收到最终答案，不作为模型失败，也不混入正式八次分母。当前主运行目录为 `runs/sol_medium_c12/`，接续 `runs/sol_medium/continuation.json`；后者保留初始原始记录和切换说明。发送总数最多比正式试验数多四次（若后续服务故障另行补齐，将额外记录）。网关返回的 `max_output_tokens` 为 null，已有响应实际输出 16,539 token，超过原请求上限。用户随后明确不要求限制 token；保留原请求以保持同轮一致，不因超出它重跑或判失败。本轮不视为严格 token 预算实验。API 的 output_tokens 包含 reasoning_tokens；报告区分推理 token 与二者差值（非推理输出），不能把全部输出用量当作最终 JSON 长度。

最小测试覆盖：48 个参考解重放、形状和 DAG 规则、初始空格/重复移除/越界/空洞、错误棋盘与格式、逆向边/回访/非最短、截断不重采。

请求采用 Responses 的 `reasoning.effort` 字段；官方模型资料见 [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)。本次使用用户指定的第三方网关，模型列表已确认含请求的型号；列表不能证明底层实际模型身份。


## 后台执行与最终验收

用户要求由脚本采样、结束后统一验收。`src/batch.py` 启动采样脚本，结束后自动重放全部完整输出并生成分析，最后写入 Git 忽略的 `runs/finalization.json`。`ready_for_acceptance` 只在三组最终试验数和每题八次记录全部齐全时出现；不完整或脚本错误写 `needs_attention`，不会伪装完成。后台日志在对应运行目录的 `console.log`。

切换独立后台执行前，工具会话中的采样进程已消失，52 个完整答案保存在 `runs/sol_medium_main/run.json`；均已保留。该阶段最多十个可能在途槽位另记客户端中断，接续文件为 `runs/sol_medium_main/continuation.json`。加上之前并发切换，客户端中断发送数的上界为十四；确认服务错误另外计数。新调度器增量保存在途槽位，后台执行继承已观察到的网关并发限制。当前后台输出目录为 `runs/sol_medium_background/`，分析生成在其 `analysis/` 子目录；正式验收后将紧凑结果提升到 `results/`。

已在进程环境设置密钥后，可在 Windows 独立后台启动（不将密钥写入命令参数文件）：

```powershell
Start-Process python -WindowStyle Hidden -ArgumentList @('experiments/sol_dag_blocks/src/batch.py', '--continue-from', 'experiments/sol_dag_blocks/runs/sol_medium_main/continuation.json', '--out', 'experiments/sol_dag_blocks/runs/sol_medium_background')
```


续跑说明：保留 `runs/sol_medium_reconnected/continuation.json` 中的 180 个完整答案，恢复权限后由独立后台脚本在 `runs/sol_medium_recovery/` 补齐剩余试验。此前流式连接重置（WinError 10054）导致调度器退出；异常捕获和有限重试已修复，九项测试通过，正式续跑前提交修复。已完成模型失败不重采；退出时未保存的在途结果另作未知的客户端中断记录。网络沙箱拒绝连接（WinError 10013）不会自动重试。
