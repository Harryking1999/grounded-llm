# 开发机使用手册

更新：2026-10-09。平台、028 和 4090 的 SSH 登录已验证；028、4090 已配置本机公钥。本轮已按平台 → login02 → 4090 完成模型、地图、评测数据与 adapter 中转。

## 1. 机器概览

| 机器 | 用途 | 使用约定 |
|---|---|---|
| 原人工智能平台 | 当前训练、评测；开发环境共享 `/zhanghanyue` | 环境变更后，从平台页面确认 SSH 端口 |
| 028：8 张 H100 80GB | 后续训练 | 10 月 12 日以后使用；崔涵需要 6–8 卡训练时协调释放资源 |
| 4090：本次识别到 5 张卡 | 推理、评测 | 预留 1–2 张卡供其他使用者使用，其余空闲卡可用 |
| login02 登录节点 | 028 连接入口及文件中转 | 仅用于登录和传输 |

[平台入口](https://172.16.78.10:32206/index.html?#/developEnv)：业务管理 → 开发环境 → SSH 查看。

## 2. SSH 连接

本地命令使用 PowerShell；登录节点命令使用 Bash。密码通过 SSH 提示输入，不存入仓库。

**原平台**：本机已配置 SSH 密钥。

```powershell
# 当前续训环境：4 张 A800，每卡 batch=1
ssh -p 41511 root@172.16.78.10
# 当前主评测环境：2 张 A100
ssh -p 35016 root@172.16.78.10
```

训练环境另有本机别名 `grounded-dev`。密钥和连接配置位于 `%USERPROFILE%\.ssh\`。

**028、4090**：本机已配置连接别名及公钥，私钥保留在本机。

```powershell
# 028：root@nv-h100-028
ssh grounded-028
# 4090：gongruochen@10.28.0.80
ssh grounded-4090
```

两个别名均经过 `zhangyue@172.16.78.36:10022`（login02）。目标机器已通过仅公钥认证的连接验证；login02 未开放公钥认证，每次连接仍需输入一次跳板密码。

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

原平台目录支持跨开发环境复用。**028、login02、4090 与平台之间无已确认的共享目录**；028 的 `/ssdwork` 为节点本地磁盘。

2026-10-09 最初在 35016、40327 合用八卡评测；随后端口改为 41511，并先尝试两卡续训。用户最终将 41511 扩为四张 A800，确认每卡 batch=1，从本轮 10 epoch 完整断点续训到 20 epoch；35016 两张 A100 改为主评测，4090 的 4 号卡辅助评测。正式安排仍见 `experiments/flamingo_map_reader/configs/blocks_kv_two_node_continuation.json`：主、辅助分别持有分片索引模 3 的余数 0/1、2，结果回到同一平台 `evaluation_half_epoch/`。旧八卡队列与两卡尝试作为历史保留。

当前训练、评测源码为 `ce8b3b7`，平台部署于 `code/blocks_kv_three_node_ce8b3b7/`，4090 位于项目目录 `code/ce8b3b7/`。续训日志为平台运行目录下 `logs/training_twenty_four_gpu_ce8b3b7.log`，状态、实际合同与来源仍在 `training_extension_20epoch/{status,config,continuation}.json`。主队列日志为 `evaluation_half_epoch/logs/queue_main_35016_three_node_ce8b3b7.log`。两卡尝试的目录整体保留为 `training_extension_20epoch_two_gpu_retired_343506c/`，最后未发布的更新未带入四卡续训；原 10 epoch final 与所有权重保留。

4090 复用评测只需 adapter，不需 optimizer 和 RNG：adapter 约 98 MB，完整续训断点约 314 MB，基座约 3.10 GB，710 道评测任务的缓存约 93 MB。4090 的 0–3 号卡供现有 vLLM 服务使用，本轮辅助评测仅用 4 号卡，显存约 48 GB。基座、冻结地图、评测缓存及所需 adapter 已转入项目目录；使用独立环境 `/home/gongruochen/grounded_llm/.venv/bin/python`，不修改公共 Conda 环境。

独立环境已对齐平台的 PyTorch `2.7.1+cu128`、Transformers `4.57.6` 和 h5py `3.16.0`，Python 为 3.13，平台为 3.11。以原 10 epoch adapter 和相同题目验证 reference、rollout 各 2 题，生成时间分别约 18.4、16.5 秒/题，正式辅助分片也已连续产出回答。两条 reference 的动作序列与平台缓存一致；两条 rollout 中一条路径一致、另一条从第三个动作起分歧，两边这两题均未到达。运行库对齐后仍有跨机器的生成差异，原因未单独定位，不声称逐 token 复现；辅助输出保留独立来源及日志。

按用户确认，文件先存入 login02，再复制至目标机。已验证跳板分别可达 `nv-h100-028` 和 4090 的 `ubuntu`，并完成 adapter 的跳板至 4090 中转，数据段约 98 MB/s。028 本轮只核对连接，仍遵守 10 月 12 日之后使用的约定。login02 需要密码，目标 4090 从跳板复制也使用其账号认证；密码仅在运行进程中使用，未写入仓库或脚本。

模型与数据经平台直接上传到 login02，再复制到 4090，避免本机上传瓶颈。平台运行目录 `transfer/` 存放中转包、后台传输脚本、`4090_bridge.log` 和 `4090_bridge_status.json`；独立传输环境为平台工作区 `.transfer-venv/`。该后台程序每 10 分钟检查新权重、同步平台已完成分片的摘要标记并回传辅助分片，发布 adapter 后才发布可评测标记，分片返回时最后写摘要。摘要标记用于跨目录复用和全队列结束判断，完整逐题结果统一在平台汇总。它负责实验文件传输，不会创建 Codex 定时唤醒；30 分钟检查仍关闭。4090 的辅助日志位于其运行目录 `evaluation_half_epoch/logs/queue_4090_aux_ce8b3b7.log`，绝对路径仅通过显式 `source_path_map.json` 对应，监督配置保持一致。

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
