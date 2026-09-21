# Blocks Step 2：文字状态加连续 token 是否改善逐步解题

按 2026-09-21 用户决定，当前执行合同为 [blocks_step2.json](configs/blocks_step2.json)。
它取代旧的 [pilot_design.json](configs/pilot_design.json) 提案：仅比较文字与文字加 token，
自报错误立即失败，暂不测十二块，不展开多模型、多 token、LoRA 或非空目标条件。

## 假设与对照

二值棋盘已是完整状态，直接经过 MLP 映射为 Qwen 输入向量，无需学习 Step 1 的 Q roadmap。
两组获得同一初始文字棋盘、规则、历史动作和自行报告的棋盘；+token 组额外获得同状态编码。
使用用户确认的 Qwen3-4B-Instruct-2507；冻结 LLM，仅训练状态报告 adapter。
报告训练只提供状态 token，要求输出棋盘，防止 adapter 靠复制文字绕过像素；测试时加回相同文字。
训练与规划存在任务切换，因此同时汇报独立棋盘读出准确率。

沿用旧 `BlocksTask` 的棋盘、形状、方向、孔洞规则与无库存限制。
降低难度只减少构造块数；不根据模型成败或求解器难度挑题。
训练、开发和测试按原始棋盘家族划分，并排除跨划分平移等价的完整及派生状态。
构造见证仅用于保证有解和训练状态采样，不进入模型输入或 loss。

## 每步协议

1. 模型生成一个包含 `shape_id, row, col, board_after` 的 JSON；`board_after` 是动作后完整棋盘。
2. 在完整 JSON 边界暂停。裁判先验证动作，再核对自报；任何错误直接失败，不纠正、不重试。
3. 双重通过后，保留相同文字输出；+token 组将已核验状态经 adapter 转成向量，追加到同一 assistant 序列。
4. 继续下一步。棋盘清空且最后一次自报正确才成功。提前 done、格式、上下文或总预算耗尽分别计数。

两组相同文字边界；初态也提供对应 token。模型已经正确报告的棋盘就是文字状态，不再追加一份裁判文字真值。
每步重新计算完整前缀，步内使用模型 KV cache；当前不实现跨步骤 cache。
生成上限按整题累计；没有暗设每步新预算、搜索工具、候选动作筛选或强制合法动作语法。
额外 token 会增加输入计算量，报告实际 prefill token 与时间，不宣称 FLOPs 完全匹配。

## 复用与运行

功能归属见 [MODULES.md](../../MODULES.md)。模型加载、adapter、loss、优化、训练、选模和 generation 均复用公共 harness。
基础数值设置继承已存在 Step 2 配置，运行时只保存实际使用字段，排除旧图任务参数。

```sh
python -m unittest discover -s experiments/state_interface_pilot/tests -v
python -m grounded_llm.harness \
  --config experiments/state_interface_pilot/configs/blocks_smoke.json \
  --model-dir "$MODEL" --model-provenance "$MODEL/step2_provenance.json" --output "$SMOKE"
python -m grounded_llm.harness \
  --config experiments/state_interface_pilot/configs/blocks_step2.json \
  --model-dir "$MODEL" --model-provenance "$MODEL/step2_provenance.json" --output "$RUN"
```

正式运行重新初始化 adapter，不沿用 smoke 权重。开发集完整报告准确率优先、答案 loss 次之、较早轮次再次之选模；
测试题不参与选择。每个输出目录保存已解析合同、源码提交、模型来源、软件版本、数据划分、权重、
原始逐步 token/文字与逐项裁判结果。已有输出目录拒绝覆盖。

多卡执行仍使用同一入口：`--blocks-phase train` 只训练与读出评测；
`--blocks-phase eval --training-run "$TRAIN" --condition text_token --shard-index 0 --num-shards 7`
读取该运行已保存的数据及选中权重，仅执行索引对应的题目。文字组将 condition 改为 text，
可以在训练过程中运行，因为不依赖权重。token 组必须通过开发集读出检查，不能仅等待训练结束。
评测先检查输入、训练及生成配置与来源一致，
分片只改变调度，不改变题目、预算、prompt 或 batch 内采样。报告生成可批处理，规划每题独立。

2026-09-21 首轮报告训练未成功，代理仍启动了规划评测；这一推进错误保留记录，相关结果不用于
判断有效状态 token 的规划价值。后续训练在开发集不达配置的完整棋盘准确率门槛时停止，不再读取测试探针或进入规划。
门槛是工程准入条件，并不保证多步规划有效。

固定小样本诊断同样复用训练器：配置 `blocks_fit.json`，使用
`--blocks-phase fit --training-run "$TRAIN"` 读取原训练划分中固定棋盘。
只做小集报告拟合与输入状态轮换检查，不启动规划、不测测试集；不将小集拟合正确率当泛化能力。

## 结果怎么读

主比较为各难度配对解题率差；同时报告合法动作数、动作和报告均正确的连续步数、报告准确率、失败类型及成本。
动作合法但自报错误仍计一次合法动作，随后立即停止；因此合法动作数与通过步骤数不混用。
配对 bootstrap 区间只描述当前测试题抽样不确定性，不代表训练随机性；本轮为单训练 seed、每题一次 greedy 试点。

如果读出不准，规划零增益不能解释为状态 token 原理无效；如果读出准但规划不改善，则支持读取与使用是不同瓶颈。
文字组已能解的容易题用于识别上限，困难题用于观察失败位置，不将跨协议历史基线直接混入当前对照。
当前执行进展只维护于 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)，结果出来后在本目录保存紧凑摘要。
