# GCML Luna v3 Blocks Baseline

状态：已完成 8 条英文测试，但该轮不是能力层面完全隔离的纯文本基线；仅作为工具暴露与状态错误的诊断记录。主 Blocks 基线见 `experiments/gcml_luna_v5_blocks/`。

## 运行合同

- 模型：`gpt-5.6-luna`
- 推理强度：`medium`
- 隔离：每条是独立 projectless black-box task；模型未获得仓库、论文、父线程或其他案例的上下文。
- 输入：GCML 官方 `tiling_order_10x10_8obj.h5` 的行 18000--18002，按已记录的官方二值网格和 8 种带方向积木形状给题；B18000、B18001、B18002 循环抽样共 8 条。
- 动作：`(shape_id,row,column)`，零起点；形状覆盖的当前像素必须全部为 1，合法动作将其清零；目标为全零网格。
- 本组保留 8-object demo 的 8-action budget。该预算是本组输入合同，不宣称所有 GCML 任务都只有 8 步。
- 模型被要求一次性先规划，并为每个尝试动作输出 `state_before`、动作、`predicted_state_after` 和一句可审计 rationale。逐字隐藏式 CoT 不在 API 可交付范围内，未伪造或重建。

## 判定定义

- **action-execution pass**：动作可解码、每一步在真实初始网格/真实后继状态上合法、8 步内清空网格。
- **trace diagnostic**：记录是否提供了 steps/state 字段，以及这些字段是否与独立裁判重算一致；不是任务 pass 的硬门槛。
- 任何真实环境中的非法动作在该案例处立即停止并判失败；其后的模型文本不挽救该案例。
- 日志中的 `verdict JSON` 是外部裁判结果，原始模型输出和完整 API transcript 均保留。

## 结果

| case | input | action-execution | trace | 主要现象 |
| --- | --- | --- | --- | --- |
| 01 | B18000 | FAIL | FAIL | 第 2 步 shape 7 覆盖零像素，立即非法 |
| 02 | B18001 | PASS | FAIL | 8 步真实动作合法并清空，但只给 actions 列表，缺少逐步账本 |
| 03 | B18002 | FAIL | FAIL | 第 1 步 shape 3 覆盖 `(2,1)` 零像素，立即非法 |
| 04 | B18000 | PASS | FAIL | 8 步真实动作合法并清空，但输出为无 rationale 的数组 |
| 05 | B18001 | FAIL | FAIL | 动作本身合法，但状态/后继预测多处错误，真实网格最后仍留 `(5,8)` |
| 06 | B18002 | FAIL | FAIL | 回答称看不到网格并转向无关 dashboard，未给动作 |
| 07 | B18001 | PASS | FAIL | 8 步真实动作合法并清空；第 1 步后继预测和第 2 步 state_before 错误 |
| 08 | B18002 | PASS | FAIL | 动作可按形状名称宽松解码并清空，但不符合要求的 steps/state/rationale schema |

- action-execution `pass@8 = 4/8 = 0.5000`（仅作 v3 诊断，不作为当前 Blocks 主基线）
- 可解析且执行成功的结果不因缺少逐步账本而额外判失败；v3 中 case 02、04、08 的动作序列因此计入 action-execution，但 trace 仅作诊断。
- exact required steps schema：2/8（case 05、07）；两条均存在状态/后继预测错误。
- 失败轴：非法动作 2/8；格式或环境偏离 1/8；动作合法但状态/转移预测导致目标失败 1/8；仅格式缺失但真实动作成功 3/8。

## 解释边界

这组结果支持“工具面暴露和较紧的输出合同会混淆 Blocks 诊断”的结论；它不能单独证明缺少状态网络是唯一原因。case 02、04、08 的真实动作成功说明可解析动作与逐步账本应分开报告；case 05 是最清楚的状态/转移错误反例，case 06 是工具隔离失败。更清晰的文本合同重跑见 v5。

## 日志

- `logs/case_01.md` ... `logs/case_08.md`：每条线程的原始英文输入、模型最终输出、可见 reasoning summary、裁判 verdict 和完整 API thread record。
- 日志与运行产物位于忽略路径，不纳入 Git 源码提交。
