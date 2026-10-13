# 开发机使用手册

更新：2026-10-13。44491 已恢复，当前为一张 A100 80GB；35016 当前拒绝连接。4090 按现有约定使用 1–4 号卡，0 号卡预留。

## 1. 机器概览

| 机器 | 用途 | 使用约定 |
|---|---|---|
| 原人工智能平台 | 44491 已恢复，当前为一张 A100 80GB；共享 `/zhanghanyue` | 35016 当前拒绝连接；使用显式 SSH 端口 44491 |
| 028：8 张 H100 80GB | 后续训练 | 10 月 12 日以后使用；崔涵需要 6–8 卡训练时协调释放资源 |
| 4090：5 张 48GB 卡 | SFT 半轮续训与 510 题串行评测 | 按续训合同使用 1–4 号卡，0 号卡预留 |
| login02 登录节点 | 028 连接入口及文件中转 | 仅用于登录和传输 |

[平台入口](https://172.16.78.10:32206/index.html?#/developEnv)：业务管理 → 开发环境 → SSH 查看。

## 2. SSH 连接

本地命令使用 PowerShell；登录节点命令使用 Bash。凭据不存入仓库。用户已授权保存跳板和 4090 口令，已用 Windows DPAPI 加密保存到本机 `%USERPROFILE%\.codex\credentials\grounded-machines.json`；后续直接读取并在进程内解密使用，不重复询问。持久连接辅助入口为同目录的 `grounded-ssh.py`，使用本项目 `.venv` 的 Python，通过 Paramiko 连接并核验现有 known_hosts；stdin 接收公开的命令／文件传输 JSON，不接收明文口令。`tmp/sft_remote.py` 只是本次运行的临时副本。

**原平台**：本机已配置 SSH 密钥。

```powershell
# 当前可用的一卡 A100 环境
ssh -p 44491 root@172.16.78.10
# 35016 当前拒绝连接，不再作为本次读取目标
# 41941已关闭并从当前运行合同禁用，不再恢复
# 原训练环境41511已完成20 epoch并关闭，不再恢复
```

本机别名 `grounded-dev` 仍记录此前的 35016，本次 44491 使用显式端口；当前 4090 续训通过 `grounded-4090` 与已保存的连接辅助入口管理。密钥和连接配置位于 `%USERPROFILE%\.ssh\`。

**028、4090**：本机已配置连接别名及公钥，私钥保留在本机。

```powershell
# 028：root@nv-h100-028
ssh grounded-028
# 4090：gongruochen@10.28.0.80
ssh grounded-4090
```

两个别名均经过 `zhangyue@172.16.78.36:10022`（login02）。目标机器已通过仅公钥认证的连接验证；login02 未开放公钥认证。交互式 SSH 会提示跳板密码；代理应使用上述已保存的本机凭据，不再向用户询问。

单独登录跳板可使用 `ssh grounded-login02`。本机项目密钥为 `%USERPROFILE%\.ssh\id_ed25519_grounded_dev`，公钥已登记到两台目标机器及跳板；跳板公钥暂不生效。

Agent 使用相同连接方式，登录后通过 `hostname` 确认主机、`nvidia-smi` 查看 GPU 占用。本次本地直连 4090 失败，跳板连接成功。

## 3. 目录与产物

| 内容 | 当前位置 |
|---|---|
| 本地源码 | `D:\Grounded_llm`；Git 远端为 `https://github.com/Harryking1999/grounded-llm.git` |
| 平台早期实验 | `/zhanghanyue/experiment/grounded_llm` |
| 平台当前地图读取工作区 | `/zhanghanyue/experiment/flamingo_map_reader`；包含 `code/`、`runs/`、`models/`、`logs/`、`.venv/` |
| 当前 K/V 重训产物 | 上述工作区 `runs/blocks_kv_restart_36c213c/`；权重位于 `training/models/`，评测位于 `evaluation_half_epoch/` |
| 20 epoch 续训产物 | 同一运行的 `training_extension_20epoch/`；新增权重位于 `models/`，与原 10 epoch final 分开 |
| 上一轮寻路、积木产物 | 上述工作区 `runs/long_f73b700_20261004/{path,blocks}/` |
| 当前基座模型 | `/zhanghanyue/experiment/flamingo_map_reader/models/Qwen2.5-1.5B-Instruct` |
| 当前积木地图 | `/zhanghanyue/experiment/grounded_llm_qmap_tree_132f5a1/runs/blocks_distance_map/tree_1000_132f5a1/best.pt` |
| 已完成同题对照、尚未接入 LLM 的续训地图 | `/zhanghanyue/experiment/grounded_llm_qmap_tree_132f5a1/runs/blocks_distance_map/tree_continue_986cb4d/checkpoints/step_015000.pt` |
| 028 基座候选 | `/ssdwork/fuzhizhang/model_base/Qwen2.5-1.5B-Instruct`；复用权限待确认 |
| 4090 公共模型 | `/opt/models`；2026-10-09 重查未发现 Qwen2.5-1.5B-Instruct，项目基座由平台经 login02 复制到项目目录 |
| 4090 项目目录 | `/home/gongruochen/grounded_llm/`，含 `code/`、`models/`、`runs/` 与 `incoming/` |
| login02 项目中转 | `/home/zhangyue/transfer/grounded_llm/`，已创建并完成 adapter 中转验证 |
| 当前无地图 SFT | 4090 的 `runs/blocks_reason_sft_bs4_50k/` 保存训练断点、新评测与迁移来源；原平台同名运行保留历史模型、逐题回答与评测。公共中转为 login02 的 `transfer/grounded_llm/llm_task_sft/` |

原平台目录支持跨开发环境复用。**028、login02、4090 与平台之间无已确认的共享目录**；028 的 `/ssdwork` 为节点本地磁盘。

SFT 启动前实查：原 35016 主机身份为 `56ldennbmqmq3-0`；2026-10-12 替换平台 44491 的主机身份为 `7pkc1o9ovj34v-0`，四张 A100-SXM4-80GB，原共享路径、虚拟环境及已完成基座评测均保留。两平台后由用户关闭。4090 主机身份为 `ubuntu`，五张卡各 49,140 MiB；SFT 三轮权重留存于 `training/checkpoint-{12500,25000,37500}/`，当前 `models/epoch_{1,2,3}/` 为对应断点的符号链接。平台三轮模型和原始评测仍在共享盘，不删除或覆盖。

2026-10-12 18:31（北京时间）按用户授权启动 4090 评测，源码 `d2d9a54`。本地已验收的基座 validation/test、前两轮 validation 完整汇总已迁入 4090；来源记录在 `runtime/migration_44491_audited_summaries.json`，明确标注历史逐题分片未迁回。队列只生成缺少的三轮 test 与第三轮 validation，共 214 个分片，不混用平台未完成的 test。该阶段现已完成，profile、PID 与日志路径保留在 `runtime/{evaluation_profile,evaluation_processes}.json`，原日志为 `evaluation_queue_4090_d2d9a54.log`。

随后用户重新提供 35016。实查主机为 `2q9cq8lsupefi-0`，四张 A100-SXM4-80GB 均空闲，没有本项目活动进程；共享路径、三轮整数模型、数据与环境完整保留。查明两机均无本轮半整数权重后，用户明确选择“先登记 35016；本轮只评已有整数权重”。因此不补训或评测 0.5、1.5、2.5 epoch，不在 35016 启动额外任务，4090 队列继续。

用户随后授权从第三轮继续训练，并明确“只用 4090”、只评固定 510 题。2026-10-12 21:18（北京时间）启动源码 `9834539` 的续训协调器，原前三轮 test 已完整完成。新的运行输出为原运行目录下 `continuation/`，profile 与进程信息为 `runtime/continuation_{profile,process}.json`；日志为 `continuation/logs/coordinator_9834539.log` 和逐段 `train_to_<step>.log`。协调器从原完整 `training/checkpoint-37500/` 恢复，每段训练后保存 `continuation/training/checkpoint-<step>/`，再用同机四卡完成该点的 510 题两种评测，输出于 `continuation/evaluation/epoch_<epoch>/test/`；完整成绩汇总到 `continuation/curve.json`。不启动 35016，不再生成 validation 评测。原模型、恢复状态与历史评测均保留。

续训正式合同为 `experiments/llm_task_sft/configs/blocks_reason_continuation.json`，延续原 cosine 末尾学习率 2e-6，完整恢复 optimizer、scheduler、RNG 与半轮样本位置。当前 `continuation/status.json` 和 `continuation/training/progress.json` 为运行状态；没有总 epoch 上限，失败或空间不足时停止并记录原因，不自动清理产物。需要在周期结束后停止时，可在 `continuation/` 创建 `stop_after_cycle`。

2026-10-13 11:03（北京时间）验收：4、4.5 点各已完成全部 42 分片与完整恢复断点，协调器已自动切到 `train_to_62500.log`，从 `continuation/training/checkpoint-56250/` 接续。累计六个评测点验收通过，动作回放、汇总与训练窗口的完整诊断保存在 `runtime/acceptance_current.json`，逐题涨跌与首轮一致性为 `runtime/acceptance_paired.json`；紧凑摘要见仓库 `experiments/llm_task_sft/results/acceptance.{md,json}`。本次复用前四点，新增审计两点，没有重新生成回答；全部操作仅在 4090，未修改训练配方。

当前 44491 的主机身份为 `4b5glpbtq0bnc-0`，一张 A100-SXM4-80GB；共享目录与原虚拟环境保留，工作区仍为 `/zhanghanyue/experiment/flamingo_map_reader/`。

2026-10-13 在该平台完成旧地图训练示范长度的 CPU 核查，原始摘要与一次性检查脚本位于当前 K/V 运行 `runs/blocks_kv_restart_36c213c/diagnostics/training_route_lengths/{summary.json,audit.py}`；未修改训练数据或占用 GPU。结论与紧凑计数归入仓库 `experiments/llm_task_sft/results/policy_comparison.{md,json}`。

同题 510 的身份核对保留在原平台 `runtime/test510_alignment.json`，4090 复用已向平台提供的同一数据目录。概率对照源码 `8d1f467`，原输出位于平台 `diagnostics/random_largest_{validation,test510}/`，紧凑摘要已保存在仓库结果目录。源代码按 Git commit 部署到 `code/llm_sft_<commit>/`；运行时路径与 PID 放在本次运行的 `runtime/`，不入 Git。实验合同见 `experiments/llm_task_sft/configs/blocks_reason_bs4.json`；以下地图实验机器分工均为历史。

2026-10-09 最初在 35016、40327 合用八卡评测；随后端口改为 41511，并先尝试两卡续训。用户最终将 41511 扩为四张 A800，确认每卡 batch=1，从本轮 10 epoch 完整断点续训到 20 epoch；35016 两张 A100 改为主评测，4090 的 4 号卡辅助评测。正式安排仍见 `experiments/flamingo_map_reader/configs/blocks_kv_two_node_continuation.json`：主、辅助分别持有分片索引模 3 的余数 0/1、2，结果回到同一平台 `evaluation_half_epoch/`。旧八卡队列与两卡尝试作为历史保留。

2026-10-09 两节点重启后，按用户确认交换分工：35016 四张 A100 续训，41511 四张 A800 主评测（两路 rollout、两路 reference）；每卡 batch=1、全局 batch=4，继续到原定 20 epoch。最新完整断点为 `training_extension_20epoch/models/checkpoint-145000/`（11.6 epoch），含 optimizer、scheduler、trainer state 和四份 RNG。重启前日志到 146221 步，最后未保存的约 1221 步从断点重跑；原 10 epoch final、所有完整 checkpoint 和已完成评测分片保留。平台与 4090 的分片余数仍为 0/1、2。

2026-10-10 用户再次重启两节点后，按“四卡节点继续训练”调整：41511 的四张 A800 从最新完整 `checkpoint-236250`（18.9 epoch）续训，35016 的两张 A100 主评测，一路 rollout、一路 reference。35016 当前环境名为 `20261010143741`，旧环境 `20261009224734` 不再作为恢复目标；41511 环境仍为 `20261009175224`。端口均不变。停机前日志 236281 步，未保存的 31 步重跑；optimizer、scheduler、trainer state 和四份 RNG 全部恢复。artifact_bridge 随主评测迁至 35016，仍每 600 秒经 login02 同步。当前日志为 `logs/training_41511_resume_fb923f8.log`、`logs/evaluation_35016_resume_fb923f8.log`，runtime profile 原位更新。以下旧机器分工和日志名称仅作为历史。

2026-10-10 按用户授权检查扩卡时，原四卡 vLLM 服务已自行退出，0–3 号卡显存均为0，无需杀进程或删除部署。预留0号卡，4090评测改用1–4号卡（两路rollout、两路reference）。主节点35016保持两卡，分片改为模3余数0；4090持有余数1/2。切换时仅停止本项目旧队列和未完成worker，完整summary保留；先同步全部完成标记与缺少的adapter，再通过现有experiment_runner启动四卡辅助队列。4090仍使用ce8b3b7评测实现，仅部署仓库现有experiment_runner与正式两节点合同；运行路径保存于忽略的runtime/auxiliary_evaluation_profile.json。artifact_bridge在35016恢复，仍每600秒同步。41511已正常退出训练（runner returncode=0），20 epoch final adapter与evaluation_ready.json均已生成；不会再恢复此轮训练。以下单卡4090规则仅作为历史。

2026-10-10 用户新增41941用于评测，SSH主机身份为`2ibgniuee46br-0`，三张A800，启动前无实验进程；原训练节点41511已由用户关闭。当前评测总计九卡：35016两卡、41941三卡、4090四卡（预留其0号卡）。正式合同原位加入`additional_evaluation`，三路模9互斥分片：main余数0/1、platform_extra余数2/3/4、4090_aux余数5/6/7/8；普通510题仅有分片索引0–7，余数8用于保持分区完整。新节点共享平台模型、数据、权重与输出目录；其忽略运行配置为`runtime/additional_evaluation_profile.json`，日志`logs/evaluation_41941.log`，状态`runner.additional_evaluation.json`与`evaluation_half_epoch/status.platform_extra.json`。可复用runner新增第三评测角色，源码与合同提交b072a2c，部署于既有code目录；4090相同runner接口同步更新，评测实现保持原条件。切换前仅停止本项目评测队列与未完成worker，完整结果复用，先同步完成标记后启动。artifact_bridge仍在35016每600秒运行；不启动原训练节点。此前mod3的分片分工均作为历史。

2026-10-10 按用户“平衡、尽快完成”要求，九张评测卡均改为按空闲情况领取reference或rollout。35016与41941的五张卡共享平台待评池，原子创建分片占位目录保证互斥；已验证41941无法重复领取35016正在评测的分片，未采用在该共享目录上失效的跨节点advisory锁。正式合同的shard_overrides将早期权重的部分辅助分片转给平台；具体区间、编号与队列读取合同，不在本手册复制参数。4090其余任务仍独立、0号卡继续预留。切换使用experiment_runner及runtime profile中的resume_running，核对后接续原九个worker，无需重跑；新增源码已部署到既有平台与4090代码目录。评测条件与文件同步入口保持原条件。

2026-10-11 用户确认平台仅剩35016。新主机身份为`5id3tvc40vs5p-0`，实测四张A100；41941拒绝连接，正式合同禁用additional_evaluation，runner拒绝启动已禁用角色。先核对本节点无评测进程，再将旧平台队列状态保存到忽略的runtime/retired_platform_state，标记额外队列retired并清理五个空的旧分片占位；完整结果保留。恢复artifact_bridge后回传4090尚未同步的完整分片。35016四卡与4090的1–4号卡按合同的新互斥分工继续剩余任务，所有卡均可领取reference或rollout；4090的0号卡预留。初次恢复时关闭旧PID接管，因为旧平台容器已退出、辅助旧队列已完成；随后按合同重新平衡积压分片时，仅重启调度进程并接管已核对的存活worker，保留正在生成的分片。使用相同experiment_runner和代码路径恢复；bridge随新合同重新启动，仍经login02中转。此前两卡平台与九卡分工仅作历史。

平台恢复源码基于已提交的 `fb923f8`，部署于 `code/blocks_restart_fb923f8/`，随后原位部署本仓库可复用调度修改；训练与评测通过 `experiment_runner` 启动。原 warmup 保持 6250 步；数据、地图、监督目标和评测实现沿用原条件。4090 保留 `code/ce8b3b7/` 路径，并原位部署相同调度接口。

平台运行目录的 `runtime/{training,evaluation,bridge}_profile.json` 保存路径和显式恢复断点，不含密码；实际进程状态分别在 `runner.training.json`、`runner.evaluation.json`，训练进展仍读 `training_extension_20epoch/logs/train.jsonl`。本次训练、主评测日志为 `logs/training_35016_resume_fb923f8.log`、`logs/evaluation_41511_resume_fb923f8.log`。恢复前的 continuation/status 副本保存在 `runtime/`，两卡尝试仍保留于 `training_extension_20epoch_two_gpu_retired_343506c/`。重启中断且没有完成摘要的评测分片由原队列保留为 `.interrupted.*` 后重做，完整分片直接复用。

4090 文件同步已改用可复用 `artifact_bridge`，运行于 41511；每 600 秒经 login02 中转新 adapter 与完成标记，再导入完整辅助分片。日志与状态分别为运行目录 `transfer/artifact_bridge_status.log`、`transfer/artifact_bridge_status.json`。独立 `.transfer-venv` 需要 Paramiko 和 filelock；本次从已安装的 ML 环境复制 filelock 3.18.0 到传输环境，未修改 ML 环境。跳板和目标密码通过交互提示进入内存及后台 stdin，不写入 profile、argv、文件或日志。

4090 复用评测只需 adapter，不需 optimizer 和 RNG：adapter 约 98 MB，完整续训断点约 314 MB，基座约 3.10 GB，710 道评测任务的缓存约 93 MB。4090 的 0–3 号卡供现有 vLLM 服务使用，本轮辅助评测仅用 4 号卡，显存约 48 GB。基座、冻结地图、评测缓存及所需 adapter 已转入项目目录；使用独立环境 `/home/gongruochen/grounded_llm/.venv/bin/python`，不修改公共 Conda 环境。

独立环境已对齐平台的 PyTorch `2.7.1+cu128`、Transformers `4.57.6` 和 h5py `3.16.0`，Python 为 3.13，平台为 3.11。以原 10 epoch adapter 和相同题目验证 reference、rollout 各 2 题，生成时间分别约 18.4、16.5 秒/题，正式辅助分片也已连续产出回答。两条 reference 的动作序列与平台缓存一致；两条 rollout 中一条路径一致、另一条从第三个动作起分歧，两边这两题均未到达。运行库对齐后仍有跨机器的生成差异，原因未单独定位，不声称逐 token 复现；辅助输出保留独立来源及日志。

按用户确认，文件先存入 login02，再复制至目标机。已验证跳板分别可达 `nv-h100-028` 和 4090 的 `ubuntu`，并完成 adapter 的跳板至 4090 中转，数据段约 98 MB/s。028 本轮只核对连接，仍遵守 10 月 12 日之后使用的约定。login02 需要密码，目标 4090 从跳板复制也使用其账号认证；密码仅在运行进程中使用，未写入仓库或脚本。

模型与数据经平台直接上传到 login02，再复制到 4090，避免本机上传瓶颈。平台运行目录 `transfer/` 存放中转包与旧 `4090_bridge.*` 记录；当前后台同步入口、状态和日志见上文 `artifact_bridge`。独立传输环境为平台工作区 `.transfer-venv/`。同步每 10 分钟检查新权重、同步平台已完成分片的摘要标记并回传辅助分片，发布 adapter 后才发布可评测标记，分片返回时最后写摘要。摘要标记用于跨目录复用和全队列结束判断，完整逐题结果统一在平台汇总。它负责实验文件传输，不会创建 Codex 定时唤醒；30 分钟检查仍关闭。4090 的辅助日志位于其运行目录 `evaluation_half_epoch/logs/queue_4090_aux_ce8b3b7.log`，绝对路径仅通过显式 `source_path_map.json` 对应，监督配置保持一致。

项目目录如下，028 目录尚未创建，4090 与 login02 目录已创建：

| 机器 | 项目目录 |
|---|---|
| 028 | `/ssdwork/zhanghanyue/grounded_llm` |
| 4090 | `/home/gongruochen/grounded_llm` |
| login02 中转 | `/home/zhangyue/transfer/grounded_llm` |

项目目录按 `code/`、`models/`、`maps/`、`data/`、`runs/`、`logs/` 分类。

平台 Python：`/zhanghanyue/experiment/flamingo_map_reader/.venv/bin/python`。4090 Conda：`/opt/miniconda3`。新机器使用独立环境，依赖安装在项目环境中。

## 4. 文件传输

源码通过 Git 同步，或将已提交版本打包传输。执行前创建本地 `tmp/` 和远端接收目录：

```powershell
git archive --format=tar --output=tmp/grounded-source.tar HEAD
scp -P 10022 tmp/grounded-source.tar zhangyue@172.16.78.36:/home/zhangyue/transfer/grounded_llm/
```

以下示例需预先创建接收目录，并替换 `run-id`、`/path/to/model` 及对应本地路径。SSH 端口参数为 `-p`，scp 为 `-P`；目录复制使用 `-r`。

**028 → 4090**：本次直连返回网络不可达，使用 login02 中转。在 login02 执行：

```bash
scp -r root@nv-h100-028:/ssdwork/zhanghanyue/grounded_llm/models/run-id /home/zhangyue/transfer/grounded_llm/
scp -r /home/zhangyue/transfer/grounded_llm/run-id gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/
```

**平台 → login02 → 4090**：平台可直接访问 login02 的 SSH 端口。在平台与 login02 分别执行，预先创建中转与接收目录：

```powershell
# 平台上
scp -P 10022 -r /path/to/model zhangyue@172.16.78.36:/home/zhangyue/transfer/grounded_llm/
# login02 上
scp -r /home/zhangyue/transfer/grounded_llm/model gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/
```

大文件可在 login02 使用 `rsync -a --partial --info=progress2` 传输，支持中断后续传。

Reader 模型迁移包含基座、adapter、配置和对应地图；续训需完整 checkpoint。数据迁移需包含软链接目标，并更新绝对路径。模型和原始数据保存在 Git 外。

## 5. 文档维护

- 机器、端口或目录变更：更新本手册对应表格。
- 模型或数据迁移：记录来源、目的地、代码提交和验证结果。
- 实验进度与下一步：更新 [项目状态页](docs/PROJECT_STATUS_AND_TODO.md)。
- 实验参数与结果：更新对应实验的 `configs/` 和报告。

Agent 在远端操作前读取本手册，完成后更新新增产物的位置。

## 6. 本地旧工作树归档

2026-10-09 按用户确认，将以下工作树整体移入 Git 忽略的 `D:\Grounded_llm\tmp\archived-worktrees\`，并将分支改为 `codex/archive-*`。提交、源文件、`runs/`、`tmp/` 和已有本地环境均保留，没有删除运行产物。

| 原位置 | 归档目录（上述路径下） | 归档分支 |
|---|---|---|
| `D:\Grounded_llm_global_context` | `step2-global-context` | `codex/archive-step2-global-context` |
| `D:\Grounded_llm_harness` | `experiment-harness` | `codex/archive-experiment-harness` |
| `D:\Grounded_llm_step2` | `step2-path` | `codex/archive-step2-path` |

这些仍是可恢复的 Git 工作树，不计入当前研究工作区。若需恢复，可用 `git worktree move` 移回原路径；原路径恢复前，不假定旧虚拟环境或记录的绝对路径仍可直接运行。文献报告和 Qwen 基线工作树本轮只核对差异，尚未删除。
