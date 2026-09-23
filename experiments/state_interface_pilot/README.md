# Previous approach：Blocks Step 2 连续 token 读出与逐步解题

本目录封存原积木棋盘 → MLP → LLM 连续 state token 路线。以下合同、运行命令和结果说明用于复现历史实验，不再是当前下一步；新的显式 roadmap 接口不会加入本目录。当前方向见[研究简述](../../docs/RESEARCH_BRIEF.md)和[项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

按 2026-09-21 当时的决定，原执行合同为 [blocks_step2.json](configs/blocks_step2.json)。
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

## 多样像素状态的报告训练

按用户补充要求，[blocks_readout.json](configs/blocks_readout.json) 覆盖不同占据率、合法移除中间态、
多个不连通区域、孤立格、带孔洞局部区域及逐格破坏后的中间态。全空和全满是训练边界样本，
不计作未见开发样本。像素读出不要求棋盘有解；后续规划题仍由原 `BlocksTask` 生成有解实例。
生成器见 `grounded_llm/blocks_readout.py`，标签始终是完整二值棋盘，无动作或隐藏分解监督。
原棋盘派生样本不跨家族划分，完全相同或平移等价状态不跨训练／开发／原规划测试集。

使用同一 harness 的 `--blocks-phase train`，该配置仅允许报告训练，不自动解题。
保存 `readout_dataset.json`、`coverage.json` 以及按状态类别分层的 `readout_validation.jsonl`；
同时记录完整棋盘、格式有效性、全零输出、逐格匹配与占据格召回的原始计数。
训练 epoch 数是本次运行预算，是否学会由自由生成读出决定。

小集两百步后仅 2/8 正确但 loss 仍下降。`blocks_fit_continue.json` 用于进一步定位是否训练不足；
`--adapter-init` 明确加载原权重，优化器重新初始化，不能称为恢复相同优化器轨迹。
多样状态训练从头初始化，不使用小集诊断权重。

## 位置读出续训诊断

小集续训最终达到 8/8 且随输入改变；多样集十轮后仍为 0/356，出现随占据率改变而输出零／一的模式。
这不支持容量不足解释，也尚未证明已学会像素位置。下一轮从同一个多样集末次权重出发：
`blocks_readout_continue.json` 继续完整报告；`blocks_readout_rows.json` 将部分报告替换为指定行的报告。
两组都从原运行读取已保存的棋盘数据，以 `--training-run` 绑定；用 `--adapter-init` 显式加载权重，优化器重新初始化。
按配置匹配每板报告数、更新次数和学习率；行报告更短，不声称 token／FLOPs 匹配。
行号只是提问，adapter 始终编码同一完整 100 像素，不接收行号、动作或正确答案。
开发集继续以完整棋盘报告选模；另保存类别平衡的训练集读出和开发集单像素翻转后的行读出诊断。
部分像素准确或行报告成功不能直接放行规划。

## 单格查询 CE 与断点续训

用户指出小集每板训练次数远多于多样集，随后指定改用单格查询重新训练。
短程结果只描述当时权重，不能比较收敛后的泛化能力。完整报告长程续训尚未启动，本轮改为
[blocks_cell.json](configs/blocks_cell.json)：重新初始化 adapter，保持冻结 Qwen 与单个状态 token，
只训练 `p(bit | token, row, col)` 的普通全词表交叉熵。只监督一个答案 bit，无 EOS、行报告、完整棋盘或 swap loss。
前缀不包含其他真实格子的答案；没有额外可训练解码器。

采样器每轮重新从既有训练棋盘抽取查询，精确均衡坐标 × 0/1；不声称每张棋盘采样次数均等，
也不声称这完全消除了覆盖率相关性。模型选取按全部开发棋盘、全部坐标的未约束下一 token 正确率，再按 CE。
同时报告占据／空格准确率、类别平衡准确率，以及逐格拼回棋盘的 exact；另报告只在 0/1 logits 间选择的诊断结果，
不与未约束输出混合。逐格重建成功不等于完整棋盘自回归生成成功，也不能直接放行原规划协议。
每次验证都读取固定训练子集，帮助区分未拟合与泛化差距。

共同平台期要求训练 loss 与开发 loss 均无明显改善，且开发主指标不再改善。
预算、门槛、检查间隔以配置为准。它是操作性停止标准，不是全局最优或读出成功的证明；
预算耗尽仍未进入平台时必须标为未确认收敛。

长程训练保存 `adapter/latest.pt`，包含 adapter、优化器、洗牌状态、已完成轮数、历史曲线和选中权重，
通过临时文件替换避免中断写入破坏上一检查点。节点中断后用同一份已提交代码和配置，在新输出目录中运行：

```sh
python -m grounded_llm.harness --config experiments/state_interface_pilot/configs/blocks_cell.json \
  --model-dir "$MODEL" --model-provenance "$MODEL/step2_provenance.json" --blocks-phase train \
  --training-run "$PREVIOUS" --resume-checkpoint "$PREVIOUS/adapter/latest.pt" --output "$NEW_RUN"
```

`--resume-checkpoint` 与 `--adapter-init` 互斥；前者恢复训练状态，后者只是加载权重重新优化。
恢复后的预算轮数包括检查点已经完成的轮数，每轮查询由轮号确定，不重新抽取已完成轮次。
小型确定性测试确认中断恢复后的参数与连续训练一致；不承诺不同 GPU／软件环境下逐位复现。

双卡使用 [blocks_cell_parallel.json](configs/blocks_cell_parallel.json)，两份持久冻结 LLM 分担同一批样本，
仅保留一个 adapter、优化器和采样器。有效 batch 与原训练相同；单卡原先分两次累积的 microbatch
改为两张卡同时各处理一半。交叉熵仍逐答案平均，再按实际样本数平均，不按设备平均。
cell 验证同样分片后按原顺序汇总。断点可从单卡切到双卡，运行时设备与分片大小单独记在配置，
不改变优化器超参数或数据抽样；浮点归约顺序可能带来微小数值差异。

## 结果怎么读

主比较为各难度配对解题率差；同时报告合法动作数、动作和报告均正确的连续步数、报告准确率、失败类型及成本。
动作合法但自报错误仍计一次合法动作，随后立即停止；因此合法动作数与通过步骤数不混用。
配对 bootstrap 区间只描述当前测试题抽样不确定性，不代表训练随机性；本轮为单训练 seed、每题一次 greedy 试点。

如果读出不准，规划零增益不能解释为状态 token 原理无效；如果读出准但规划不改善，则支持读取与使用是不同瓶颈。
文字组已能解的容易题用于识别上限，困难题用于观察失败位置，不将跨协议历史基线直接混入当前对照。
该路线的已确认结果见[读出报告](results/readout.md)，原配置和运行记录继续保留。
