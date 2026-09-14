# Qwen thinking：匹配 Sol 的路径与积木基线

检验官方 Qwen3 不同参数量模型在现有完整计划任务上的成功率。复用
`sol_dag_blocks` 已固定的 suite、英文提示词、任务规则及精确裁判，不重新选题。
研究合同见 `configs/thinking.json`。该实验不训练状态模块，也不向模型提供裁判工具。

模型使用官方 Qwen3 同代 4B/8B/32B thinking；不以 Base、额外 SFT/GRPO 或
2507 版本替代。采样设置依据 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B)。
GPT 的 medium 与 Qwen thinking 没有等价计算量保证；输出预算包含推理和最终答案。

## 接口

运行环境需 SGLang、PyTorch、Transformers，当前验证环境记录在实际运行
`run_config.json`。模型路径、端口、GPU 和输出位置均由 CLI 提供；源码不绑定机器。
模型若缺失可通过 ModelScope `snapshot_download` 下载官方模型到用户指定目录。
`src/download.py --model-id Qwen/Qwen3-4B --model-dir "$MODEL_PATH" --staging-dir "$LOCAL_STAGING"`
提供可续传的下载入口；共享盘小文件写入较慢时，将临时分片留在本地盘，再保存完整权重。

```sh
python experiments/qwen_path_blocks/src/serve.py \
  --config experiments/qwen_path_blocks/configs/thinking.json \
  --model-path "$MODEL_PATH" --gpus "$GPU_IDS" --port "$PORT" --cache-dir "$CACHE_DIR"

python experiments/qwen_path_blocks/src/run.py \
  --config experiments/qwen_path_blocks/configs/thinking.json \
  --suite "$SUITE_PATH" --model-id Qwen/Qwen3-4B --model-path "$MODEL_PATH" \
  --endpoints http://127.0.0.1:31004 --out "$OUTPUT_DIR" --smoke-only
# 确认正式批次的前两项正常后，同命令去掉 --smoke-only 并加 --resume。
# 多副本服务可在 --endpoints 后传多个 URL。
```

每个槽位的原始输出、推理文本、最终答案、token 用量、终止原因与裁判结果立即落盘。
首两项本身属于正式采样，不是额外筛选；任务失败和预算截断保留，不重采。
传输错误归档并保持槽位未完成，显式 `--resume` 可补齐；已完成槽位不会重复调用。
只有全部派发请求结束后才重放和汇总。主要成功率只由环境执行结果决定：严格 JSON、
最终状态字段和逐步棋盘自报都是诊断项，不构成失败。可无歧义恢复的 JSON 动作／路径数组
会在送入同一裁判前规范化；无法形成完整最终答案的截断仍失败。报告逐次成功率、pass@8、
截断、格式恢复来源、非法动作和合法前缀状态报告；保留原始 thinking，但不将其当作模型内部机制的直接证据。

共享结果根目录由本次运行指定在用户的 `experiment/grounded_llm` 下，独立于 OPD。
原始模型可只读复用既有目录；不引入 OPD 的训练脚本或运行流程。

本轮结果见 [验收报告](results/report.md)：path 按最终 256 节点双向图、blocks 按
16k 输出预算汇报，包含准确率、可见思考案例及截断前错误重分类。

若 16,384 token 条件出现大量积木截断，可使用 `configs/thinking_32768.json` 追加匹配批次。
该合同将服务上下文设为官方 Qwen3 运行时接受的 40,960 token，并将并发降低以保留更长 KV cache；其结果与 16K 条件分开报告。

若目标是尽量排除输出预算作为积木失败原因，使用 `configs/blocks_40000.json` 配合
`src/select_suite.py --conditions blocks8 blocks12` 生成 256 槽的 blocks-only suite。40,000
输出 token 加上最长 548-token 输入仍低于 Qwen3 的 40,960-token 上限；该合同将每个服务
的并发降至 2，以便为长 KV cache 留出空间。它是以积木为中心的独立预算条件，不能与
16K 或全套 32K 条件合并。

`configs/path256_bidirectional_16k.json` 是独立的 256 节点稀疏双向最短路条件。使用
`src/prepare_path_suite.py --config ... --out "$SUITE_PATH"` 冻结一个 16 题、每题 8 次的
suite；图为固定的 3-regular 连通图，节点行和邻居顺序按题目打乱。它保留 16,384 token
上限，截断计失败，不能与原 32 节点 DAG 或 40K 积木条件合并。

## Sol API 同题试点

2026-09-14 用户授权通过 `https://jarodfund.xyz` 测试 `gpt-5.6-sol`，沿用上述
path256 无向图与 16k 条件。已保存待运行的 suite 和提示词，并确认 suite 与三个 Qwen
验收批次完全相同、邻接表对称、全部参考路径通过原裁判。发现网关参数未严格受控后，
用户要求先试点几次；这次授权只推进小批试点，没有启动完整批次。

[接口探针摘要](results/sol_api_compatibility.json)记录了阻碍：Responses 的
`max_output_tokens`、Chat Completions 的 `max_completion_tokens` 和 `max_tokens`
均请求 32 token，却分别报告输出 302、203、203 token，正常完成而非截断。
Responses 还返回空的输出上限，以及不同于请求的 temperature/top_p 和额外系统指令。
这不能证明该网关也一定忽略 16k，但不足以确认预算与提示条件匹配；三个探针不是 path
样本，不进入成功率或 pass@8。原始响应位于 Git 忽略的 `runs/sol_path256_16k/`，不含请求密钥。

试点参数见 [Sol 试点合同](configs/sol_path256_pilot.json)，使用原 suite 的前四题，
每题一次，沿用原提示词、内容解码器和无向最短路裁判。原始请求与完整响应保存于
`runs/sol_path256_16k/pilot/`。因首轮多个请求返回网关错误，随后仅对前两个服务失败槽位
各补试一次，并降低并发；实际配置、错误和接续输出保存在同根目录的 `pilot_retry/`。
有模型答案的槽位不重采；服务错误单独报告。试点结果不并入严格匹配预算的 Qwen 结果，
也不估计 pass@8。

**试点结果：获得的 3 条完整答案全部合法到达且为最短路。** 共尝试 6 次 API 请求，
其中 3 次返回网关服务错误，没有模型答案；`path_undirected_256_03` 仍无有效输出。
原始首轮和补试记录均保留，具体计数、返回参数和同题 Qwen 背景见
[试点摘要](results/sol_path256_pilot.json)。

| 题目 | 起点→终点 | 模型步数／最短步数 | 输出 token（含推理） | 其中推理 | 请求耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| 00 | 4→98 | 9／9 | 964 | 876 | 39.9 秒 |
| 01 | 51→92 | 8／8 | 1,012 | 932 | 106.2 秒 |
| 02 | 211→219 | 8／8 | 1,469 | 1,389 | 42.1 秒 |

三条答案实际均未超过请求预算，但返回上限仍为空，temperature/top_p 仍与请求不同，
且附带额外系统指令；不能据此宣称预算已被执行或与 Qwen 严格匹配。有效输出合计
3,445 token，服务失败请求的用量未知。所有完成答案已重放通过，提交给 API 的题目提示词
与冻结的 Qwen 提示词一致，无工具调用。题目 01 的 Qwen 三档历史结果均为 0/8，Sol
此次为 1/1；这表明该题可被当前 Sol 接口解决，不足以估计模型总体差距或稳定成功率。
单并发补试成功，但没有隔离并发、时间和服务负载，不能将先前错误确定归因于并发。

```powershell
# 在进程环境中设置 GND_API_KEY；输出目录必须尚不存在。
python experiments/qwen_path_blocks/src/run_api_pilot.py --config experiments/qwen_path_blocks/configs/sol_path256_pilot.json --out runs/sol_path256_16k/pilot
```

## 离线验收与截断归因

`src/acceptance_audit.py` 读取已完成原始运行，复用任务裁判，输出紧凑诊断证据：

```sh
python experiments/qwen_path_blocks/src/acceptance_audit.py \
  --runs "$RUN_4B" "$RUN_8B" "$RUN_32B" \
  --conditions blocks8 blocks12 --out "$AUDIT_PATH"
```

Path 将 `--conditions` 改为 `path_undirected_256`。原始准确率不改变；实际触顶单独标记。
触顶前最终答案的已确定动作若非法，优先归非法；合法路径前缀若已不可能最短，归非最短。
未完成的数字不猜测，未定案思考不自动当作答案，剩余截断仅称原因未决。完整报告中的
思考行为摘要需人工检查原始 `reasoning_text`，不能从内容判分派生文件中恢复。
