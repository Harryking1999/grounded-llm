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

当前状态：迁移脚本和正式合同已建立，等待本轮完整批次结果。

若 16,384 token 条件出现大量积木截断，可使用 `configs/thinking_32768.json` 追加匹配批次。
该合同将服务上下文设为官方 Qwen3 运行时接受的 40,960 token，并将并发降低以保留更长 KV cache；其结果与 16K 条件分开报告。
