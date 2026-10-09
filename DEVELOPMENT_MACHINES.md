# 开发机使用手册

更新：2026-10-09。平台、028 和 4090 的 SSH 登录已验证；028、4090 已配置本机公钥。文件传输示例未执行。

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
# 当前训练环境：4 张 A100
ssh -p 35016 root@172.16.78.10
# 当前评测环境：4 张 A800
ssh -p 40327 root@172.16.78.10
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
| 上一轮寻路、积木产物 | 上述工作区 `runs/long_f73b700_20261004/{path,blocks}/` |
| 当前基座模型 | `/zhanghanyue/experiment/flamingo_map_reader/models/Qwen2.5-1.5B-Instruct` |
| 当前积木地图 | `/zhanghanyue/experiment/grounded_llm_qmap_tree_132f5a1/runs/blocks_distance_map/tree_1000_132f5a1/best.pt` |
| 028 基座候选 | `/ssdwork/fuzhizhang/model_base/Qwen2.5-1.5B-Instruct`；复用权限待确认 |
| 4090 公共模型 | `/opt/models`；本次未发现 Qwen2.5-1.5B-Instruct |

原平台目录支持跨开发环境复用。**028、login02、4090 与平台之间无已确认的共享目录**；028 的 `/ssdwork` 为节点本地磁盘。

新机器的建议项目目录如下，**尚未创建**：

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

**平台 → 4090**：通过本地中转。在本地 PowerShell 执行，预先创建 `./models/`：

```powershell
scp -P 35016 -r root@172.16.78.10:/path/to/model ./models/
scp -o ConnectTimeout=60 -J zhangyue@172.16.78.36:10022 -r ./models/model gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/
```

大文件可在 login02 使用 `rsync -a --partial --info=progress2` 传输，支持中断后续传。

Reader 模型迁移包含基座、adapter、配置和对应地图；续训需完整 checkpoint。数据迁移需包含软链接目标，并更新绝对路径。模型和原始数据保存在 Git 外。

## 5. 文档维护

- 机器、端口或目录变更：更新本手册对应表格。
- 模型或数据迁移：记录来源、目的地、代码提交和验证结果。
- 实验进度与下一步：更新 [项目状态页](docs/PROJECT_STATUS_AND_TODO.md)。
- 实验参数与结果：更新对应实验的 `configs/` 和报告。

Agent 在远端操作前读取本手册，完成后更新新增产物的位置。
