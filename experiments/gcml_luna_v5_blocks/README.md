# GCML Luna v5 Blocks Text-Only Baseline

状态：已完成 8 条独立、隔离提示的英文纯语言测试；本组是当前 Blocks 主基线。

## 运行合同

- 模型：`gpt-5.6-luna`
- 推理强度：`medium`
- 输入：GCML 官方 `tiling_order_10x10_8obj.h5` 行 18000--18002 的 10x10 二值网格，循环抽样 8 条。
- 每条是独立 projectless black-box task。提示词明确声明打印出的网格是完整输入，并禁止浏览器、屏幕、文件、终端、代码、网页搜索和其他工具调用。
- 输出合同只要求可解析 JSON `actions` 数组：每个动作包含 `shape_id,row,col,rationale`；逐步 state 字段可选，不作为 pass 的硬门槛。
- 外部裁判按 GCML 形状掩码执行动作；任何非法动作立即停止并判失败。最多 8 个动作，目标为全零网格。
- rationale 是公开、简短、可审计的动作理由，不是逐字隐藏式 CoT；API 不提供隐藏 CoT，本日志不重建它。

## 结果

| case | input | actions | legal | solved | tools observed |
| --- | --- | ---: | --- | --- | --- |
| 01 | B18000 | 8 | PASS | PASS | no |
| 02 | B18001 | 7 | PASS | PASS | no |
| 03 | B18002 | 8 | PASS | PASS | no |
| 04 | B18000 | 8 | PASS | PASS | no |
| 05 | B18001 | 8 | PASS | PASS | no |
| 06 | B18002 | 8 | PASS | PASS | no |
| 07 | B18001 | 8 | PASS | PASS | no |
| 08 | B18002 | 8 | PASS | PASS | no |

- action-execution `pass@8 = 8/8 = 1.0000`
- 非法动作：0/8；超出 8 步：0/8；格式不可解析：0/8。
- 这组没有产生可定位的状态维护、转移预测或规划失败；它说明当前三个官方轮廓在清晰文本合同下对 Luna 偏简单，不能支持“需要状态网络”的增益主张。

## 工具隔离说明

Codex projectless 线程在平台层仍可能看到工具定义，因此提示词禁止工具不是能力层面的绝对撤销。本组 8 条返回记录均未出现 `commandExecution`、`mcpToolCall` 或浏览器标记，故可作为“观测到的无工具运行”。上一轮 v3 Blocks 的 case 06 确实调用了浏览器/环境工具并返回无关 dashboard；它保存在 `experiments/gcml_luna_v3_blocks/`，仅作为隔离失败的诊断记录，不与本组结果合并。

## 日志

- `logs/case_01.md` ... `logs/case_08.md`：原始英文输入、最终输出、可见 reasoning summary、工具使用检查、外部 verdict 与完整 API thread record。
- 日志与运行产物位于 Git 忽略路径；README 保留为紧凑的正式摘要。

## 后续

先增加独立且更长的官方轮廓/组合实例，或改用真正不提供工具面的文本 API，再做状态账本、单步转移和规划对照；不应从本组 8/8 推断跨任务能力。
