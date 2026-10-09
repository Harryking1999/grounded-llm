# 开发机使用手册

本文件是 Grounded LLM 的机器、连接和产物位置索引，后续直接更新本文件。研究进度与下一步仍以 [PROJECT_STATUS_AND_TODO](docs/PROJECT_STATUS_AND_TODO.md) 为准；运行参数以各实验 `configs/` 为准。本手册不授权启动新实验、迁移正在运行的任务或扩大 GPU 使用。

最近核实：**2026-10-09，北京时间**。下文区分实际检查、用户给定约定和建议目录。端口、空闲 GPU 和挂载可能变化，不能把这次快照当作永久承诺。此次只检查连接、目录、网络与 GPU 信息；未启动训练、修改远端环境、搬运模型或配置新的登录密钥。

## 1. 机器与使用约定

| 环境 | 连接入口 / 账号 | 用途与约束 | 本次核实 |
|---|---|---|---|
| 本地开发机 | Windows，`D:\Grounded_llm` | 权威源码工作树、文档与紧凑结果摘要 | 已检查；其他任务的未提交文件须保留 |
| 原人工智能平台：训练环境 | `172.16.78.10:35016`，`root`；平台账号 `zhanghanyue` | 当前既定训练；共享存储 `/zhanghanyue` | SSH 密钥登录成功；页面对应 `20261009105647`，`zhangyue-node02`，4×A100 80GB |
| 原人工智能平台：评测环境 | `172.16.78.10:40327`，`root` | 当前既定评测；同一个共享存储 | SSH 密钥登录成功；页面对应 `20261009073105`，`ai-a100v1-07`，4×A800 80GB |
| 新渠道登录节点 | `172.16.78.36:10022`，`zhangyue` | 进入 028 的跳板，也是已验证的文件中转点；不在这里运行 GPU 训练 | 密码登录成功；主机 `login02.ngpu.com`；家目录 `/home/zhangyue` |
| 028 训练节点 | 在登录节点执行 `ssh root@nv-h100-028` | 用户约定 **2026-10-12 以后**才使用算力；8×H100，与崔涵协调，他可能需要 6–8 卡，任务须能让出资源 | 第二跳登录成功；内部地址 `192.168.181.28`；8×H100 80GB；检查时几乎无显存占用 |
| 4090 推理机 | `10.28.0.80:22`，`gongruochen`，通过登录节点进入 | 用户明确澄清：**留出 1–2 张卡给别人，其余可用**；用于推理/评测，实际用卡数按空闲与任务需求决定 | 经跳板密码登录成功；主机名 `ubuntu`；家目录 `/home/gongruochen`；GPU 查询返回 5 张 4090，驱动报告每卡 49140 MiB |

4090 检查时 0–3 号卡各占约 44 GB，4 号卡显存为 0。这只是当时的观测，**没有预约任何卡**。使用时留出 1–2 张卡给别人，其余空闲卡可按任务需要使用；仍须核实当时占用，不能结束别人的进程。028 检查时空闲也不代表 10 月 12 日后可以独占。

用户提供的信息称 028 可能租用至年底；期限没有独立核实。节点本地磁盘的长期保留不能依赖这项预期，重要自有 checkpoint 应在获准迁移后另存一份。

平台入口：[人工智能平台](https://172.16.78.10:32206/index.html?#/developEnv)。在“业务管理 → 开发环境 → SSH 查看”读取当前映射；重建环境后端口可能变化。上述两行节点与端口已在页面对应核对，不能按环境名称猜端口。开发环境存在剩余时长，登录失败时先查看运行状态、剩余时长与 SSH 映射；不要自行克隆、续期或停止环境。

## 2. 连接方法与认证

密码只使用用户私下提供的凭据，不写进本文件、脚本、命令行参数、Git 或运行日志。SSH 客户端在 Windows 已有 `ssh.exe` 和 `scp.exe`。本次新机器通过交互式密码登录，没有保存密码或向服务器增加公钥。

### 2.1 原人工智能平台

本地已有 SSH 配置：`%USERPROFILE%\.ssh\config` 中的 `grounded-dev` 指向 35016；本地身份文件为 `%USERPROFILE%\.ssh\id_ed25519_grounded_dev`。这些是本机私有配置，不复制进仓库。

```powershell
# 本地 PowerShell：只读连接检查
ssh -o BatchMode=yes -o ConnectTimeout=8 grounded-dev 'hostname; whoami; pwd'
ssh -o BatchMode=yes -o ConnectTimeout=8 -p 40327 root@172.16.78.10 'hostname; whoami; pwd'
```

本机对 `172.16.78.10` 的配置已指定上述密钥，所以第二条无需另加 `-i`。其他本地机器需要自行指定其已授权身份文件。现有 `grounded-step2` 别名仍指向旧端口 38430，**本次没有检查该入口，不作为当前入口使用**。

### 2.2 028：先登录，再进入计算节点

```powershell
# 本地 PowerShell，交互输入入口账号密码
ssh -p 10022 -o ConnectTimeout=60 zhangyue@172.16.78.36
```

```bash
# 以下在 login02 的 Linux shell 执行
hostname
ssh root@nv-h100-028

# 进入 028 后确认目标
hostname
whoami
pwd
```

本次 login02 → 028 无需再次输入密码；这只说明登录节点现有身份可用，**不代表本地持有同一身份**。因此不要默认本地 `ssh -J ... root@nv-h100-028` 也能免密成功，更不要复制登录节点的私钥来解决它。未来如要给本地 agent 配置计算节点直跳身份，需由用户/管理员授权。

用户提供的连接还包含可选代理转发：

```powershell
ssh -p 10022 -R 55557:localhost:7890 zhangyue@172.16.78.36
```

`-R` 把登录节点的 55557 转发到本地电脑的 7890，依赖本地代理正在监听；普通 SSH 登录和内网传文件不需要它。本次没有启用或测试此代理。这个转发不建立共享存储，也不自动给 H100 节点提供代理；H100 上的 `localhost` 是 H100 自己。不要把端口绑定扩大为所有网卡或把该代理默认写入全局配置。

### 2.3 4090：已验证的本地跳板连接

```powershell
# 本地 PowerShell：依次输入登录节点密码、4090 密码
ssh -o ConnectTimeout=60 -J zhangyue@172.16.78.36:10022 gongruochen@10.28.0.80 'hostname; whoami; pwd'
```

也可先按 2.2 登录 login02，再执行：

```bash
ssh gongruochen@10.28.0.80
```

本地直连 `ssh gongruochen@10.28.0.80` 本次在 SSH banner/密钥交换之前被服务端关闭，尚未到验证密码的阶段。不要把这个现象解释为密码错误；优先使用已验证的跳板路径。

可将下列**非秘密配置模板**合并进本机 SSH config；不要覆盖已有配置。本次没有自动安装这些别名。

```sshconfig
Host grounded-login02
    HostName 172.16.78.36
    Port 10022
    User zhangyue
    ConnectTimeout 60
    ServerAliveInterval 30

Host grounded-4090
    HostName 10.28.0.80
    User gongruochen
    ProxyJump grounded-login02
    ConnectTimeout 60
    ServerAliveInterval 30
```

首次连接新主机时检查目标和主机密钥；保留 `known_hosts`，不要用全局关闭主机密钥验证来消除问题。密码认证需要交互式终端；仅在已配置有效密钥时使用 `BatchMode=yes`。agent 可在 SSH 的隐藏密码提示中使用已授权凭据，不能将密码放在 `sshpass -p`、URL、脚本或可回显的输入中。无凭据的后续会话应让用户在终端完成登录，不擅自建立长期访问权限。

## 3. 存储边界与当前产物

**只有已确认挂载同一存储的原平台环境可以直接复用同一路径。** 新渠道的 028、login02、4090 与原平台没有已确认的共享目录。

- 原平台两个入口都挂载 `storage → /zhanghanyue`、`ssdwork → /ssdwork/zhanghanyue`，文件系统类型为 GPFS。当前项目主要使用 `/zhanghanyue/experiment/`。换平台开发环境时先检查挂载，不必再复制这些大文件。
- 028 的 `/ssdwork` 是 `/dev/nvme0n1` 上的 ext4 本地磁盘，不是原平台的 `/ssdwork/zhanghanyue`。028 上本次没有 `/zhanghanyue` 或 `/opt/models`。
- login02 的 `/home/zhangyue` 位于自身 `/home` 的 XFS 文件系统；其 `/ssdwork` 本次是根文件系统上的空目录，不共享 028 的 `/ssdwork`。028 的文件必须先拉取，不能在 login02 上直接按同名路径读取。
- 4090 的公共模型目录 `/opt/models` 存在，公共 Conda 在 `/opt/miniconda3`。本次没有核实其磁盘备份策略；不得把“公共目录”理解成与平台或 028 共享。

### 3.1 已存在并核实的路径

| 内容 | 位置 | 说明 |
|---|---|---|
| 权威源码 | 本地 `D:\Grounded_llm`；远端 `origin` 为 `https://github.com/Harryking1999/grounded-llm.git` | 集成分支 `main`；任务分支和提交按 [AGENTS](AGENTS.md) 管理 |
| 原平台早期项目根 | `/zhanghanyue/experiment/grounded_llm` | 早期实验运行与 Qwen3 模型，不能把它当作所有新运行的默认根 |
| 当前地图读取工作区 | `/zhanghanyue/experiment/flamingo_map_reader` | `code/` 为代码快照，`runs/` 为原始运行，`logs/` 为日志，`models/` 为基座，`.venv/` 为环境 |
| 当前分离 K/V 训练代码快照 | 上述工作区 `code/blocks_kv_restart_36c213c/` | 运行绑定快照；不是本地权威工作树，评测可能使用其他匹配的代码提交 |
| 当前分离 K/V 运行 | 上述工作区 `runs/blocks_kv_restart_36c213c/` | `training/models/` 是训练产物，`evaluation_half_epoch/` 是评测分片，`data/manifest.json` 是数据入口 |
| 当前冻结积木地图 | `/zhanghanyue/experiment/grounded_llm_qmap_tree_132f5a1/runs/blocks_distance_map/tree_1000_132f5a1/best.pt` | 现有训练配置中的 `map_source`；不要用其他新地图静默替换 |
| 当前基座模型 | `/zhanghanyue/experiment/flamingo_map_reader/models/Qwen2.5-1.5B-Instruct` | 当前训练配置中的 `model_source` |
| 当前 reader Python | `/zhanghanyue/experiment/flamingo_map_reader/.venv/bin/python` | 链接到 `/opt/conda/bin/python`；平台解释器为 Python 3.11.13；不能把这个 venv 原样搬到新机器 |
| 上一轮完整轨迹数据与模型 | 上述工作区 `runs/long_f73b700_20261004/{blocks,path}/` | 各有 `data/manifest.json` 和 `training/models/`；不是本轮训练的新权重 |
| 当前积木原始 HDF5 | `/zhanghanyue/experiment/grounded_llm_qmap_tree/data/raw/tiling_order_10x10_8obj.h5` | 当前数据 manifest 指向此文件；本次从 manifest 定位，未重新检查内容 |
| 028 基座候选 | `/ssdwork/fuzhizhang/model_base/Qwen2.5-1.5B-Instruct` | 目录与 `config.json` 存在；属于他人目录，仅定位，完整性及复用许可仍待确认，不能写入 |
| 4090 公共基座 | `/opt/models` | 已看到 Qwen2.5-3B-Instruct、Qwen3-8B、Qwen3-32B 等；本次列表没有 Qwen2.5-1.5B-Instruct，不能用 3B 替代当前 1.5B 实验 |

当前运行数据不是自包含目录：`blocks_kv_restart_36c213c/data/trajectories` 是链接，实际指向 `blocks_ffn_failure_95aae52/data/trajectories`；manifest 还记录 `source_manifest`、`prepared_data_source`、`q_checkpoint`、`official_data` 等绝对路径。停止旧训练不意味着这些依赖可以删除。迁移前应检查所用 manifest 的记录与链接，按实际加载依赖搬运；不得只复制看起来“最新”的目录。

其他历史 Q-map 目录仍在 `/zhanghanyue/experiment/grounded_llm_qmap_*`。需要具体旧权重时从对应实验结果与配置定位，不凭目录修改时间选模型。当前完整产物位置应随运行变更更新上表，实验结论只在实验报告与项目状态页维护。

### 3.2 新机器建议目录：尚未创建

| 机器 | 建议项目根 | 使用前必须核实 |
|---|---|---|
| 028 | `/ssdwork/zhanghanyue/grounded_llm` | 本次该目录不存在；10 月 12 日以后先协调目录所有权、磁盘配额和持久性，再创建 |
| login02 中转 | `/home/zhangyue/transfer/grounded_llm` | 容量足够且不会影响登录节点；只临时存放待传文件 |
| 4090 | `/home/gongruochen/grounded_llm` | 使用共享账号的命名约定、容量与权限；不得覆盖该账号已有工作 |

新项目根建议统一用 `code/`、`models/`、`maps/`、`data/`、`runs/<run-id>/`、`logs/`、`incoming/`。优先将自有微调权重放入项目 `models/`，公共 `/opt/models` 用于读取现有基座；未经约定不要向公共目录发布或覆盖模型。这些是建议，第一次落实后把本节改成真实路径。

## 4. 文件传输：按已验证的网络走

| 来源 → 目标 | 当前证据 | 选择 |
|---|---|---|
| 本地 → 平台两环境 | SSH 密钥成功 | `scp -P <当前端口>`；平台两环境之间通常无需复制共享文件 |
| 本地 → login02 | SSH 密码成功 | `scp -P 10022` |
| 本地 → 4090 | `ssh -J` 已实际登录成功 | `scp -J zhangyue@172.16.78.36:10022` |
| login02 → 028 | SSH 成功，login02 有 scp/rsync | 从 login02 拉取/推送，第二跳使用现有认证 |
| login02 → 4090 | SSH 密码登录成功，有 scp/rsync | 在 login02 中转 |
| 028 → 4090 | SSH 返回 `Network is unreachable` | 先拉到 login02，再送到 4090；不能默认直传 |
| 平台 → 新渠道 | 未建立可用直传路径；login02 → 平台 35016 的短时 TCP 探测未成功 | 先平台 → 本地，再本地 → 新渠道 |

这里验证的是 SSH 登录/网络路径，**尚未执行实际 scp/rsync 或大文件传输**。首次迁移先传一个很小的自有文件并确认接收路径可写，再估计容量与时间；大型数据/模型迁移须另获用户授权。下列为可复用示例，不是此次已执行的操作；路径中的 `checkpoint-X`、`run-id`、GPU 编号须按任务替换。

### 4.1 代码：Git 或小型源码包

源码只从权威仓库和指定提交分发。远端能访问 GitHub 时可以 clone/fetch，不能为图方便把正在运行的工作树自动 pull 到新版本。离线时本地打包已提交源码：

```powershell
# 本地；tmp/ 已被忽略。HEAD 是要分发的已提交代码，不含未提交修改
New-Item -ItemType Directory -Force tmp | Out-Null
git archive --format=tar --output=tmp/grounded-source.tar HEAD
git rev-parse HEAD
scp -P 10022 tmp/grounded-source.tar zhangyue@172.16.78.36:/home/zhangyue/transfer/grounded_llm/
```

接收目录应先按 3.2 确认可写并创建。在 login02 再用 `scp` 送到 028 的自有项目根；解压到新的代码快照目录，不覆盖正在使用的源码。在运行目录记下完整源码 commit 和正式配置路径；`git archive` 没有 `.git`，不能在解压目录里把 `git rev-parse` 当作来源凭据。不要打包整个用户目录、`.ssh`、`.env` 或其他项目。

### 4.2 本地与原平台

```powershell
# 上传小型源码包：接收目录须先存在
scp -P 35016 tmp/grounded-source.tar root@172.16.78.10:/zhanghanyue/experiment/flamingo_map_reader/scratch/

# 下载一个已发布的 checkpoint 目录；checkpoint-X 替换为真实名称
New-Item -ItemType Directory -Force models/platform-reader | Out-Null
scp -P 35016 -r root@172.16.78.10:/zhanghanyue/experiment/flamingo_map_reader/runs/blocks_kv_restart_36c213c/training/models/checkpoint-X models/platform-reader/
```

Windows `scp` 的端口参数是大写 `-P`，`ssh` 是小写 `-p`。复制目录时明确接收父目录，避免多套一层同名目录。下载当前仍在写入的 checkpoint 会得到不完整文件；按当前实验已有 `evaluation_ready.json` 发布规则选择完成产物。不要自行追加另一套生命周期 gate。

### 4.3 028 训练 → 4090 推理：登录节点中转

如果选择打包，在 028 上只打包已完成并整理好的自有模型和匹配地图。例如下列 `run-id` 目录需事先按第 5 节放齐权重、配置和必要资源；原基座若目标机没有，还需单独迁移。`incoming/` 须先存在。

```bash
# 在 028 执行；打包文件放在源目录之外，避免打包自身
tar -chf /ssdwork/zhanghanyue/grounded_llm/incoming/reader-run-id.tar -C /ssdwork/zhanghanyue/grounded_llm models/run-id maps/run-id
```

```bash
# 在 login02 执行。源文件是已完成、自有的迁移包；两个接收目录先按约定创建
scp root@nv-h100-028:/ssdwork/zhanghanyue/grounded_llm/incoming/reader-run-id.tar /home/zhangyue/transfer/grounded_llm/
scp /home/zhangyue/transfer/grounded_llm/reader-run-id.tar gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/incoming/
```

在 4090 确认接收目录和包内容后，解压到独立迁移目录；检查 `tar -tf <包>`，避免覆盖已有文件，再用 `tar -xf <包> -C <新目录>` 解压。按第 5 节重定位加载路径，不改原机产物。

目录较大或链路会中断时，可在 Linux login02 用双方已有的 rsync 续传：

```bash
# 不带 --delete；尾部 / 表示复制目录内容，保留源文件
rsync -a --partial --info=progress2 root@nv-h100-028:/ssdwork/zhanghanyue/grounded_llm/models/run-id/ /home/zhangyue/transfer/grounded_llm/run-id/
rsync -a --partial --info=progress2 /home/zhangyue/transfer/grounded_llm/run-id/ gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/run-id/
```

028、login02、4090 都已找到 rsync。原平台本次没有找到 rsync，因此平台迁移默认用 scp/打包，不假设 Windows 或平台安装了 rsync。028 文件需要先由 login02 拉取；不要让 028 向 `login02:22` 主动推送，本次该端口探测返回连接拒绝。

### 4.4 平台模型 → 4090：本地中转

```powershell
# models/platform-reader/ 中只放此次授权迁移的模型
scp -o ConnectTimeout=60 -J zhangyue@172.16.78.36:10022 -r models/platform-reader gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/incoming/
```

逆向搬回本地：

```powershell
New-Item -ItemType Directory -Force models | Out-Null
scp -o ConnectTimeout=60 -J zhangyue@172.16.78.36:10022 -r gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/runs/run-id models/4090-run-id
```

传输前看源大小和接收可用空间（Linux 用 `du -sh <源>`、`df -h <接收父目录>`）。不要重传已经确认相同来源的基座；不要假设同名模型就是相同版本。接收后检查所需文件、加载路径和一次获准的小型加载/推理。源文件先保留，中转文件只在确认迁移完成且允许清理后删除。

## 5. 搬模型与数据时必须带什么

| 目标 | 需要的内容 | 不应误解 |
|---|---|---|
| 冻结 reader 推理/评测 | 相同基座及 tokenizer/config、reader `adapter.pt` 与其配置、匹配的 Q-map/graph 资源、所需数据和源码提交 | reader adapter 不是完整语言模型；不能只有一个 `.pt` 就独立推理 |
| 精确续训 | 完整 checkpoint：adapter、optimizer、scheduler、trainer state、各 rank 的 RNG 文件；正式合同、同源地图与数据 | 推理用 final 不一定能续训；更换卡数或训练设置须使用实验已有迁移接口并另行明确研究条件 |
| Q-map 复用/续训 | 对应地图 checkpoint、模型定义、配置与所需训练数据；续训按该实验实现保留必要状态 | Q-map 较小可以优先搬，但不能用重训结果冒充原来的冻结地图 |
| 数据/轨迹迁移 | manifest、实际轨迹文件、依赖的地图/原始文件及必要资源 | 复制软链接不会复制链接目标；manifest 中的绝对路径也不会自动重写 |

先用 `ls -l <目录>` 和 `readlink -f <链接>` 定位实际依赖。为便于迁移，可对选定的自有目录使用 `tar -h` 解引用打包，或使用 `rsync -aL` 复制实际内容；解引用会增加体积，先检查链接没有指向未经授权的其他目录。不能广泛解引用整个共享用户目录。

异机新运行应保留原配置和 manifest 的来源，在新运行本地副本中明确重定位路径，并核对具体加载入口支持的路径参数。不要直接改平台正在运行的 manifest，不要用遍历目录的盲目字符串替换来修复路径。相同数据缓存、相同冻结地图和相同权重仍须可追溯；重训属于新运行，需重新获得实验授权和独立记录，不能因传输麻烦自动重训。

## 6. 环境、GPU 与 agent 工作流程

4090 登录提示明确要求：**不要在 Conda base 安装包，每位使用者创建独立环境**。公共 Conda 在 `/opt/miniconda3`；028 已看到 `/ssdwork/miniconda3/bin/python3`。不同机器有不同 Python/CUDA/驱动，跨机器重新建立任务专用环境，使用仓库实际依赖与实验要求；不复制 venv，不改共享 base，不因连接检查安装包。

agent 每次接手按下面顺序处理：

1. 读取 [AGENTS](AGENTS.md)、本手册与唯一项目状态页；确认本次只是检查、传文件、评测还是训练。接入机器的权限不等于启动高成本实验的权限。
2. 本地 `git status --short`、`git branch --show-current`；保留无关修改。选择当前任务分支、源码提交、已有正式合同和明确输出目录。
3. 按第 2 节登录，先 `hostname`、`whoami`、`pwd`，确认目录/挂载和读写权限。新平台端口先在页面核对，不能用旧端口成功连接就推定它是目标环境。
4. 获准运行时才检查 GPU 与现有进程，例如 `timeout 10 nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader` 和 `timeout 10 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`。查询若阻塞，不反复堆叠；先检查已有日志，报告无法核验占用。
5. 按协调结果选卡，并通过 `CUDA_VISIBLE_DEVICES`/实验已有 GPU 参数明确限制。`CUDA_VISIBLE_DEVICES=4` 后程序看到的 `cuda:0` 对应物理卡 4。4090 留出 1–2 张卡给别人，其余空闲卡可使用；028 在约定日期后仍需协调。不能仅改可见卡数就静默修改正式训练合同。
6. 进入绑定的代码目录和专用环境，使用实验现有入口。训练需可恢复且可让出资源；有人要求用卡时只处理本任务已确认的进程，保留 checkpoint 与日志，不对共享机器执行 `pkill python` 或全局清理。
7. 完成获准操作后更新产物位置与研究记录，说明实际执行的验证。不要自行启动定时监测、远端自动恢复、推送或对外通知。

## 7. 每次变更更新哪里

| 发生的变化 | 更新位置与最小内容 |
|---|---|
| 新机器、端口、账号、挂载或使用约定变化 | **本文件**第 1–3 节：日期、真实入口、实测路径、约束及未核实项；密码/密钥留在私有本机配置 |
| 模型、数据、代码快照搬到新地点 | **本文件**第 3 节增加实际目的地与原位置、是否保留源、来源 commit；运行本地配置记录具体路径 |
| 新正式运行/中断/恢复/结束 | [项目状态页](docs/PROJECT_STATUS_AND_TODO.md) 写决策相关状态、机器、输出根和下一步；运行细节留在忽略目录的日志/配置 |
| 实验参数或评测条件变化 | 对应 `experiments/<study>/configs/` 与实验 README；本手册只链接，不复制参数表 |
| 得到结果或发现负结果 | 对应实验 `results/` 的紧凑摘要/报告；原始产物留在 Git 外 |
| 只做 SSH smoke | 本文件对应验证行更新日期和成功/具体失败，足够即可；不另建带日期的新手册 |

迁移记录最小格式：`产物名称 / 来源机器与绝对路径 / 目标机器与绝对路径 / 源码 commit 与配置 / 日期 / 已复制、已加载或待验证 / 源是否保留`。单个 checkpoint 的精确路径写在对应实验记录中，本文件维护稳定项目根和关键产物入口。能靠 commit、配置和路径确定来源时，不另建哈希/manifest 系统。

当前待补齐：028 自有项目目录及其持久性/配额；028 候选基座的完整性与复用许可；4090 当前实验的 1.5B 基座；首次小文件和实际模型传输；迁移后的依赖路径、环境及加载验证。后续核实后原位更新，不能把建议目录改写成已存在事实。
