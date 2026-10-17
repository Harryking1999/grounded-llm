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

## 按任务逐项加难

用户进一步要求“几乎全对也不行，希望被难住”。合同见 [difficulty_ladder.json](configs/difficulty_ladder.json)，入口为 `src/escalate.py`。每档先完整测五题；成功率高于一半则继续升档，否则冻结同档十道独立新题确认，确认组成功率仍不高于一半才记录找到困难条件。原先“首次错误即停”的运行记录和全部答卷保留，但不再以偶发错误停止加难。任务间轮换调度，最多四个并发请求，每任务最多一个在途请求。

原初筛 `runs/five_task_sol_difficulty/` 已验证含 S/Z 十二／十六块、4×4 打乱八／十六步偏易，三阶打乱八步已返回题也全对。新确认策略在独立的 `runs/five_task_sol_challenge/` 启动，从二十块、三阶十二步打乱、4×4 二十四步打乱继续；九盘汉诺塔与 512 节点寻路各补完整档。新种子与初始档位见正式配置。原初筛不再派新题，其已有长请求正常落盘，避免阻塞其他任务。

非法动作、未到目标、无法解析和预算截断分别保留；计划成功但格式不合规仍按解出计，严格格式另计。服务失败补调用，不重新采样错误答案。确认题必须与已测状态不同，不挑选已知失败题；适应性初筛与十题确认都不足以推断精确总体成功率。配置档位全部通过时如实记作范围耗尽，不声称已找到困难条件。

- 积木复用用户此前确认的 10 种形状：保留 L3/I3、去掉二格条、加入四个方向的 S/Z 四格；从 12 次放置起测。形状和编号只读取原正式配置。构造沿用原含 Z 条件的连通棋盘；此次同时改变形状集与构造方式，不将结果差异归因于单一因素。
- 魔方升至三阶，六面 3×3 数字，允许六面的正／逆四分之一转；沿用独立动作前后示例，以打乱长度逐档增加。
- 华容道升至 4×4，0 仍为空格、方向仍移动空格；不加动作示例或冗余规则。
- 汉诺塔从九盘继续增加，目标为长距离区间；不再因八盘两例错误停止升档。
- 寻路保留无向无权三度图与原动作格式，增加节点数，每个起点选择其最远目标。

五种任务均继续寻找明显困难的条件，分别报告初筛与独立确认的结果；旧题与新尺寸、形状集的结果保持分开。

三阶魔方、4×4 华容道与含 Z 积木只保存回放通过的可解参考方案及长度上界，不假设它是最短方案。寻路与汉诺塔仍使用精确最短距离。它们是寻找失败点的适应性小样本，不能据此估计总体成功率；旧 200 题及其 32B 同题比较保持独立。

4×4 另补[距离下界压力条件](configs/puzzle15_distance_stress.json)：只接收 Manhattan 距离下界至少 40 的可解棋盘，每个动作最多使此距离减少 1，故任何有效解至少 40 步。它补足“打乱很长但实际离目标近”的风险；不把下界或参考解发给模型。该条件单独保存在 `runs/puzzle15_distance_stress/`，同样先五题初筛、必要时十道独立题确认，不混入原始成功率。

若二十四块档位耗尽仍未确认困难条件，队列接上[密集积木条件](configs/blocks_dense_stress.json)的二十八块，保持十形状和 10×10 尺寸。原始构造中记录形状与占格数；密度接近上限时可行性会改变实际形状比例，不把差异归因于块数一个因素。二十八块的五题已通过生成与参考解回放；三十块虽能生成单题，批量检查仍失败，三十二块也未通过当前有界构造，因此两档均不派题、不记作模型失败。

三度寻路图在 2048 节点仍全部解出，因此另补[长路径条件](configs/path_distance_stress.json)：均匀带标号树，精确最短距离至少 64，保持原无向无权规则和文字模板；BFS 参考解只供裁判使用。树图改变了拓扑及路径长度，结果单列于 `runs/path_distance_stress/`，不归因于节点数一个因素。仍采用五题初筛与十题独立确认。

```powershell
python -u -m experiments.task_text_baselines.src.escalate --config experiments/task_text_baselines/configs/difficulty_ladder.json --out runs/five_task_sol_challenge
```
