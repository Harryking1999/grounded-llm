# 开发机使用手册

更新：2026-10-09。平台、028 和 4090 均已成功登录；文件传输示例尚未实际执行。

## 1. 用哪台机器

| 机器 | 用途 | 使用约定 |
|---|---|---|
| 原人工智能平台 | 当前训练、评测；开发环境之间共享 `/zhanghanyue` | 换环境后从平台页面查看新的 SSH 端口 |
| 028：8 张 H100 80GB | 后续训练 | 10 月 12 日以后使用；崔涵需要 6–8 卡训练时协调让出 |
| 4090：本次查到 5 张卡 | 推理、评测 | 留出 1–2 张卡给别人，其余空闲卡可用 |
| login02 登录节点 | 连接 028、给两台新机器中转文件 | 不在这里跑训练 |

[平台入口](https://172.16.78.10:32206/index.html?#/developEnv)：业务管理 → 开发环境 → SSH 查看。

## 2. 怎么连接

以下命令在本地 PowerShell 执行。密码在 SSH 提示时输入，不写入仓库。

**原平台**：本机已经配置好密钥。

```powershell
# 当前训练环境：4 张 A100
ssh -p 35016 root@172.16.78.10
# 当前评测环境：4 张 A800
ssh -p 40327 root@172.16.78.10
```

本机 `grounded-dev` 别名也可连接训练环境。密钥和连接配置在 `%USERPROFILE%\.ssh\`。

**028**：先连登录节点，再进入计算节点。

```powershell
ssh -p 10022 zhangyue@172.16.78.36
```

```bash
# 在登录节点 login02 上执行，当前第二跳无需再输入密码
ssh root@nv-h100-028
```

**4090**：从本地通过登录节点连接，依次输入两个账号的密码。

```powershell
ssh -o ConnectTimeout=60 -J zhangyue@172.16.78.36:10022 gongruochen@10.28.0.80
```

Agent 也使用这些连接方式；登录后用 `hostname` 确认机器、`nvidia-smi` 查看用卡情况。本地直连 4090 本次失败，使用上面的跳板方式即可。

## 3. 东西放在哪里

| 内容 | 当前位置 |
|---|---|
| 本地源码 | `D:\Grounded_llm`；Git 远端为 `https://github.com/Harryking1999/grounded-llm.git` |
| 平台早期实验 | `/zhanghanyue/experiment/grounded_llm` |
| 平台当前地图读取工作区 | `/zhanghanyue/experiment/flamingo_map_reader`；下面有 `code/`、`runs/`、`models/`、`logs/`、`.venv/` |
| 当前 K/V 重训产物 | 上述工作区 `runs/blocks_kv_restart_36c213c/`；权重在 `training/models/`，评测在 `evaluation_half_epoch/` |
| 上一轮寻路、积木产物 | 上述工作区 `runs/long_f73b700_20261004/{path,blocks}/` |
| 当前基座 | `/zhanghanyue/experiment/flamingo_map_reader/models/Qwen2.5-1.5B-Instruct` |
| 当前积木地图 | `/zhanghanyue/experiment/grounded_llm_qmap_tree_132f5a1/runs/blocks_distance_map/tree_1000_132f5a1/best.pt` |
| 028 上找到的基座候选 | `/ssdwork/fuzhizhang/model_base/Qwen2.5-1.5B-Instruct`；位于他人目录，复用前确认 |
| 4090 公共模型 | `/opt/models`；本次没看到当前实验需要的 Qwen2.5-1.5B-Instruct |

原平台共享目录可以跨开发环境复用。**028、login02、4090 不共享这些目录**；028 的 `/ssdwork` 是节点本地磁盘。

新机器建议按以下位置整理，**目前还未创建**：

| 机器 | 项目目录 |
|---|---|
| 028 | `/ssdwork/zhanghanyue/grounded_llm` |
| 4090 | `/home/gongruochen/grounded_llm` |
| login02 中转 | `/home/zhangyue/transfer/grounded_llm` |

项目目录统一放 `code/`、`models/`、`maps/`、`data/`、`runs/`、`logs/`。平台当前 Python 是 `/zhanghanyue/experiment/flamingo_map_reader/.venv/bin/python`；新机器单独建环境，4090 的 Conda 在 `/opt/miniconda3`，不要在公共 base 里装包。

## 4. 怎么传文件

代码用 Git，或本地打包已提交的源码（先建好本地 `tmp/` 和远端接收目录）：

```powershell
git archive --format=tar --output=tmp/grounded-source.tar HEAD
scp -P 10022 tmp/grounded-source.tar zhangyue@172.16.78.36:/home/zhangyue/transfer/grounded_llm/
```

下面示例中的接收目录先创建，`run-id` 和 `/path/to/model` 换成实际目录。SSH 端口用 `-p`，scp 端口用 `-P`，复制文件夹加 `-r`。

**028 → 4090**：028 直连 4090 本次报网络不可达，先由 login02 拉取，再传到 4090。以下两条都在 login02 执行：

```bash
scp -r root@nv-h100-028:/ssdwork/zhanghanyue/grounded_llm/models/run-id /home/zhangyue/transfer/grounded_llm/
scp -r /home/zhangyue/transfer/grounded_llm/run-id gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/
```

**平台 → 4090**：通过本地中转。以下在本地 PowerShell 执行，`./models/` 先创建：

```powershell
scp -P 35016 -r root@172.16.78.10:/path/to/model ./models/
scp -o ConnectTimeout=60 -J zhangyue@172.16.78.36:10022 -r ./models/model gongruochen@10.28.0.80:/home/gongruochen/grounded_llm/models/
```

文件较大时，login02 上可以用 `rsync -a --partial --info=progress2` 替代 `scp -r`，方便断线后续传。

搬 reader 模型时带上基座、adapter、配置和对应地图；续训再带完整 checkpoint。当前平台数据有软链接和绝对路径，搬数据时连实际文件一起搬，并调整新位置的路径。模型和原始数据放在 Git 外。

## 5. 以后更新哪里

- 换机器、端口或目录：直接更新本手册的表格。
- 搬模型或数据：记下来源、目的地、对应代码提交和是否验证成功。
- 实验进度与下一步：更新 [项目状态页](docs/PROJECT_STATUS_AND_TODO.md)。
- 实验参数与结果：更新对应实验的 `configs/` 和报告。

Agent 每次远端工作先读这份手册；操作完成后，把新增产物的真实位置补回来。
