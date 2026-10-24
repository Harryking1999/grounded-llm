# 开发机使用手册

本文件只维护连接、存储、环境与迁移方法。实验进度见 [项目状态页](docs/PROJECT_STATUS_AND_TODO.md)，运行记录、日志和结果归对应实验目录。

## 1. 机器概览

| 机器 | 连接入口 | 资源约定 |
|---|---|---|
| 人工智能平台 | `root@172.16.78.10`，已登记 SSH 端口 `35016`、`40048`、`40621`、`43848` | GPU 与端口随开发环境配置变化，以平台页面为准 |
| 028 | `root@nv-h100-028`，经 login02 | 八张 H100 80GB；崔涵需要 6–8 卡时协调资源 |
| 4090 | `gongruochen@10.28.0.80`，经 login02 | 五张扩容 48GB 卡；本项目使用 1–4 号卡，0 号卡预留 |
| login02 | `zhangyue@172.16.78.36:10022` | 仅用于登录和文件中转 |

[平台入口](https://172.16.78.10:32206/index.html?#/developEnv)：业务管理 → 开发环境 → SSH 查看。

平台评测连接别名使用 `35016`。端口登记只说明连接入口，具体用途由本次任务与相应 runtime 确定。

## 2. SSH 连接与认证

本地使用 PowerShell，远端使用 Bash。

```powershell
ssh -p 35016 root@172.16.78.10
ssh -p 40048 root@172.16.78.10
ssh -p 40621 root@172.16.78.10
ssh -p 43848 root@172.16.78.10
ssh grounded-login02
ssh grounded-028
ssh grounded-4090
```

- `grounded-028`、`grounded-4090` 均经 login02；`grounded-dev` 指向 35016；访问平台前仍应核对已登记端口。
- 本机 SSH 配置与 known_hosts 位于 `%USERPROFILE%\.ssh\`；项目私钥为 `id_ed25519_grounded_dev`。
- 跳板和 4090 的已授权凭据通过 Windows DPAPI 加密存于 `%USERPROFILE%\.codex\credentials\grounded-machines.json`，后续直接读取，不重复询问。
- 连接辅助入口为同目录 `grounded-ssh.py`，使用本项目 `.venv` 的 Python。通过 stdin 传入公开命令或文件传输请求，凭据仅在进程内解密，不写入仓库、命令行或日志。
- 028 公钥登录使用本机项目密钥；跳板采用已保存的认证。连接时核验 known_hosts，操作前用 `hostname`、`nvidia-smi` 确认机器与资源。

**028 访问 login02 使用内网 `192.168.181.253:10022`。**外部 `172.16.78.36` 在 028 无路由，内网 22 端口未监听；内网入口的主机公钥须与已核验的外部入口一致。

## 3. 目录与产物

| 机器 | 项目根目录 | Python |
|---|---|---|
| 本地 | `D:\Grounded_llm` | `.venv\Scripts\python.exe` |
| 平台 | `/zhanghanyue/experiment/flamingo_map_reader` | `.venv/bin/python` |
| 028 | `/ssdwork/zhanghanyue/grounded_llm` | `.venv/bin/python` |
| 4090 | `/home/gongruochen/grounded_llm` | `.venv/bin/python` |
| login02 | `/home/zhangyue/transfer/grounded_llm` | 文件中转目录 |

| 共用资源 | 位置 |
|---|---|
| 028 基座模型（只读复用） | `/ssdwork/fuzhizhang/model_base/Qwen2.5-1.5B-Instruct` |
| 028 Qwen3-32B（只读复用） | `/ssdwork/fuzhizhang/model_base/Qwen3-32B`；完整 BF16 权重、配置与 tokenizer 已核对 |
| 平台、4090 项目基座 | 各项目根目录下 `models/Qwen2.5-1.5B-Instruct` |
| 4090 公共模型区 | `/opt/models`；使用前确认所需模型是否存在 |
| 平台积木地图工作区 | `/zhanghanyue/experiment/grounded_llm_qmap_tree_132f5a1` 为现有八形状地图来源；12块新条件使用 `/zhanghanyue/experiment/grounded_llm_blocks12_qmap`，节点临时盘缓存数据，产物与源码归档持久保存 |
| 平台早期实验工作区 | `/zhanghanyue/experiment/grounded_llm` |
| 平台汉诺塔／二阶魔方 Qmap | `/zhanghanyue/experiment/grounded_llm_puzzle_qmaps`；`code/current/` 保存当前已提交源码，`runs/` 与 `logs/` 用于新生成物；复用平台项目的 `.venv/bin/python`；完成的旧实验归档见第 7 节 |
| 4090 二阶魔方 Qmap | `/home/gongruochen/grounded_llm`；提交快照放 `code/`，迁入的小型精确数据放 `data/`，产物放 `runs/` 与 `logs/`；使用该项目 `.venv/bin/python`，0 号卡继续预留 |

项目目录按 `code/`、`models/`、`maps/`、`data/`、`runs/`、`logs/` 分类。具体 run、checkpoint、数据版本、PID 与日志位置由实验 README 和忽略的 runtime 配置维护，本手册不逐次追加。

平台 Qmap 在共享盘写入长时间阻塞时，可以使用节点 `/tmp/` 保存源码快照、数据缓存及活动运行镜像，完整断点、采样器状态与阶段结果再回存各任务的持久根。具体活动路径与回存入口见对应实验报告，不得仅凭持久目录未更新就重复启动训练。节点停止会丢失 `/tmp/`；恢复时以最后完整回存的断点接续，并如实说明未回存的曝光缺口。

本地日常阅读入口为 `tasks/<task>/`；选定 Qmap 的本地副本放在各任务 `Qmap/checkpoints/`（Git 忽略），来源登记在同目录 `selection.json`。远端原运行目录不随本地入口整理移动。

平台 `/zhanghanyue` 可跨其开发环境复用。028 的 `/ssdwork` 为节点本地磁盘；028、login02、4090 与平台之间没有已确认的共享目录。

各机器使用项目独立环境，不修改公共 Conda 环境或共享基座。依赖版本由实验配置和对应部署说明记录。

4090 的 `gongruochen` 账号存在独立用户硬配额（当前查询为 100GB），操作前同时检查 `quota -s`。训练完整存档以 028 为准，4090 的 `runs/map_sft_baselines/models/` 仅作受限推理缓存；其占用、自动清理与重连规则见实验 [README](experiments/map_sft_baselines/README.md) 和正式[缓存策略](experiments/map_sft_baselines/configs/bridge_retention.json)。

## 4. 文件传输与迁移

源码通过 Git 同步，或传输已提交版本的归档。先创建接收目录，再执行：

```powershell
git archive --format=tar --output=tmp/grounded-source.tar HEAD
scp -P 10022 tmp/grounded-source.tar zhangyue@172.16.78.36:/home/zhangyue/transfer/grounded_llm/
```

028 与 4090 间使用 login02 中转。以下为在 login02 执行的示例，路径应按实际产物替换：

```bash
scp -r root@nv-h100-028:/ssdwork/zhanghanyue/grounded_llm/models/run-id /home/zhangyue/transfer/grounded_llm/
scp -r /home/zhangyue/transfer/grounded_llm/run-id gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/
```

平台向 login02 传输使用 SSH 端口 10022。大文件可使用 `rsync -a --partial --info=progress2` 续传；自动同步由对应实验的 bridge 管理，不在本文件记录队列轮次或同步日志。

迁移时保留基座、接口权重、配置与匹配的地图；续训需完整恢复状态。数据包含软链接时须迁移其目标并更新路径。推理权重完整传输后才能发布评测就绪标记；原始模型、数据与运行文件均在 Git 外。

## 5. 文档维护边界

- 机器、连接端口、存储根目录或认证方法改变时，原位修改对应条目。
- 实验参数放在实验 `configs/`；启动、恢复、迁移来源、验证计数与逐次结果放在独立结果文件或忽略的 runtime，实验报告仅保留当前结论与入口。
- 当前研究状态与下一步只维护在 [项目状态页](docs/PROJECT_STATUS_AND_TODO.md)。
- 本手册不记录训练 step、epoch、分片计数、运行时间线、逐次日志或实验结论。

## 6. 本地历史工作区的恢复

日常源码仅使用 `D:\Grounded_llm`。旧 worktree 的实体检出、虚拟环境和缓存已移除；源码历史保存在第 8 节的既有归档引用，不依赖旧目录存在。

旧工作树中 Git 外的结果与材料保存在 `D:\Grounded_llm_archive\worktrees\`，保留各自的 `runs/`、`outputs/`、`tmp/` 层次：

| 产物子目录 | 对应源码引用（前缀均为 `refs/archive/2026-10-17/`） |
|---|---|
| `blocks-multigoal` | `codex/blocks-multigoal` |
| `wormi-reading` | `codex/wormi-reading` |
| `experiment-harness` | `codex/archive-experiment-harness` |
| `step2-global-context` | `codex/archive-step2-global-context` |
| `step2-path` | `codex/archive-step2-path` |
| `qwen32b-path120` | `codex/qwen32b-path120` |

没有独有运行材料的旧工作树只保留 Git 历史，不创建空产物目录。复现时按原配置与提交导出源码，并把历史绝对路径映射到归档位置。

## 7. Qmap 历史归档

| 位置 | 内容 |
|---|---|
| 本地 `D:\Grounded_llm_archive\qmap\` | 清理前的 Qmap 工作区、后续已完成负结果和小检查分项归档；根目录及各归档的 README 说明布局 |
| 平台 `/zhanghanyue/experiment/grounded_llm_puzzle_qmaps_archive/20261016_cleanup/` | 完整历史 `code/`、`runs/`、`logs/` 和传输包；内部绝对软链接已随归档重定位 |

归档位于当前工作区之外，恢复某次运行时，按其 `source_commit` 在独立目录检出源码，并使用同一归档运行的 `config.json`、数据与 checkpoint。元数据内保留的旧绝对路径属于历史记录，恢复时应指向实际归档位置。

## 8. Git 分支清理的恢复入口

旧本地分支已保存为 `refs/archive/2026-10-17/<原分支名>`；整合前全部引用还保存在仓库外 `D:\Grounded_llm_archive\git\grounded-llm-before-cleanup-20261017.bundle`，不提交进源代码仓库。恢复时先检查引用，再创建新分支，例如：

```powershell
git show refs/archive/2026-10-17/codex/cube-metric-trials --stat
git switch -c codex/recover-cube refs/archive/2026-10-17/codex/cube-metric-trials
```

本地独有源码按上述归档引用恢复，Git 外材料见第 6、7 节。已归档的远端历史也可用对应 `archive/2026-10-17/` 标签定位。
