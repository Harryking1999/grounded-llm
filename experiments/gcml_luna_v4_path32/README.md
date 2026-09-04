# GCML Luna v4 32-Node Path Baseline

状态：已完成 8 条独立、隔离的英文纯语言测试；path32 按当前修订合同验收。

## 运行合同

- 模型：`gpt-5.6-luna`
- 推理强度：`medium`
- 隔离：每条是独立 projectless black-box task；模型未获得仓库、论文、父线程或其他案例的上下文。
- 输入：GCML 官方 32-node 图邻接表，节点编号从 0 开始；8 组 start/goal 与官方图一致。
- 模型一次性规划完整路径，并为每个尝试移动输出 `state_before`、`(from,to)`、`predicted_state_after` 和一句可审计 rationale。
- 本组不施加动作步数上限。这里的 `pass@8` 中的 8 是 8 条独立采样，不是动作步数；最短性单独由官方邻接表 BFS 判定。
- 逐字隐藏式 CoT 不在 API 可交付范围内；日志保留模型可见 rationale、reasoning summary 和完整 API transcript，不伪造或重建隐藏内容。

## 判定定义

- **execution pass**：每一步沿官方无向边，动作链最终到达该案例 goal。
- **optimal**：实际动作数等于独立 BFS 的最短距离。
- **reported goal correct**：模型的 `final_node` 字段与案例 goal 一致。该字段错误不改变动作链是否实际到达目标，但作为目标判断/输出一致性问题单独记录。
- 任意非法边会立即终止并判失败；本组没有出现非法边。

## 结果

| case | start -> goal | moves | shortest | execution | optimal | final_node 字段 |
| --- | --- | ---: | ---: | --- | --- | --- |
| 01 | 26 -> 1 | 8 | 7 | PASS | FAIL | correct |
| 02 | 14 -> 30 | 3 | 3 | PASS | PASS | correct |
| 03 | 3 -> 19 | 7 | 6 | PASS | FAIL | correct |
| 04 | 0 -> 22 | 3 | 3 | PASS | PASS | correct |
| 05 | 28 -> 2 | 7 | 7 | PASS | PASS | correct |
| 06 | 19 -> 29 | 5 | 5 | PASS | PASS | WRONG (reported 1) |
| 07 | 30 -> 13 | 7 | 6 | PASS | FAIL | WRONG (reported 1) |
| 08 | 26 -> 1 | 9 | 7 | PASS | FAIL | correct |

- execution `pass@8 = 8/8 = 1.0000`
- shortest-path rate `= 3/8 = 0.3750`
- reported `final_node` correct：6/8 = 0.7500
- complete state/action trace consistency：8/8；非法边：0/8

## 解释边界

path32 的动作执行在这 8 条样例上没有暴露状态维护或单步转移失败；非最优主要是规划/路径选择问题。case 06、07 的动作链已到达目标但 `final_node` 自报为 1，属于目标判断或输出字段维护问题，不应把它误记为环境执行失败。该结果也说明 path32 当前难度偏低，后续若要区分规划能力，应增加更长或更接近的 start/goal 对，并保持同一图、同一提示和同一验收器。

## 日志

- `logs/case_01.md` ... `logs/case_08.md`：每条线程的原始英文输入、模型最终输出、可见 reasoning summary、裁判 verdict 和完整 API thread record。
- 日志与运行产物位于忽略路径，不纳入 Git 源码提交。

## 旧运行说明

此前一轮 path32 线程被错误提示词施加了 8-move 上限，未作为本组证据保存；本目录只保存去掉该上限后的 v4 线程。
