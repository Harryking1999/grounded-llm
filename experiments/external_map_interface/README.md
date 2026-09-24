# 显式 cognitive map 接口

## 当前问题与边界

本目录是新的独立实验入口。图任务先复用 Step 1 的 Q/V 地图，逐步向 LLM 提供**候选动作的 learned-map 距离**。没有把 Q 压成 token，也没有训练 LLM 解码 Q。图与积木旧 Step 2 的代码、配置和历史结果留在原目录。

图的候选分数为 `||Q[current] + V[action] - Q[goal]||₂`。`V[action]` 是 learned-map 中动作的位移，不是动作本身。每次真实执行之后，从环境获取实际新节点，重新构造当步 prompt。图的动作目录按有向边定义，本身包含目标节点；因此当前图接口把真实候选终点作为动作语义给所有条件，learned-map 条件只额外提供距离。真实图最短路只在评测端使用，绝不进入 prompt 或 Q-map 读入模块。

`src/q_map.py` 只加载 Q/V 与计算候选距离；`src/transitions.py` 管合法动作和实际状态更新；`src/interface.py` 构造每步上下文；`src/planner.py` 调用模型与解析选择；`src/evaluate.py` 执行逐步评测。每步重新发送当前图和当步候选，避免把历史向量追加到上下文。图接口只露出标量距离，不露出完整或短 Q 向量。

## 图实验入口

`configs/path32_smoke.json` 只用于检查双起终点的逐步调用链。输入为 Step 1 同一 case 目录下的 `inputs.npz` 与 `map.npz`；可使用已训练的 128 维 Q/V，不需要额外 adapter。模型路径与运行 backend 是运行参数；有 SGLang 服务时传 `--endpoint`，在节点上直接加载模型时用 `--backend transformers`。产物写入 Git 忽略的 `runs/`。`plain`、`reasoning` 不读 Q/V；`distance` 在同一图和候选动作上增加 learned-map 距离。每步原始回答都会留档；非法动作、输出截断和步数上限分别计数。256 token 的初试使两个 thinking 条件都在首步截断；1024 token 仍只是诊断预算，path32 的结果不能与正式基线比较。

正式逐步对照使用 [path256_distance.json](configs/path256_distance.json) 和[原 path256 合同](../qwen_path_blocks/configs/path256_bidirectional_16k.json)的同一冻结题集：同一 256 节点图、16 个起终点、每题 8 次。`src/train_graph_map.py` 在该图上复用 Step 1 的局部 Q/V 更新，固定训练末轮，不按规划成绩挑地图；原有三张随机 256 节点图的 Q/V 不适用于这张图。`src/evaluate_path256.py` 分别运行 Qwen3-4B 非 thinking、thinking、thinking 加候选地图距离。三组每步都看到同一图、已执行路径和排除已访问节点后的合法动作，每次执行后用环境实际后继更新当前状态。距离条件额外得到 `||Q当前+V动作−Q目标||₂`。程序按原始寻路裁判核验实际路线，不把真实最短距离传给模型。

输出预算按用户确定的**单步**范围执行：每次动作决策最多生成配置中的 token 数，包含思考；不限制整条轨迹累计输出。采样参数与原 path256 合同一致，终止、截断和 token 用量逐步存档。由于旧基线一次生成完整路线而这里是逐步反馈，三组均重新运行，历史分数仅作背景。

本图已训练的 128 维地图在原始高维 Q 空间的距离秩相关为 0.575694；一步转移 MSE 为 `7.16e-12`，后继识别为 100%。对全部 65,280 个有序起终点，只按候选 learned-map 距离取最小值，44,530 个选择落在最短路方向（68.21%）。这是地图质量诊断，不是 LLM 成绩。训练摘要与 Q/V 存在开发节点忽略路径 `runs/external_map_interface/path256_local128_dd2e5f2/`。

正式三组各 128 条逐步轨迹已完成并重放裁判验收。thinking 条件大量在单步输出预算处截断，地图组未观察到成绩提升。数值、局限与下一步建议见 [path256 显式距离报告](results/path256_distance.md)。

```bash
python -m experiments.external_map_interface.src.evaluate \
  --config experiments/external_map_interface/configs/path32_smoke.json \
  --inputs runs/cml_step1_exploration/local128/official32/inputs.npz \
  --map runs/cml_step1_exploration/local128/official32/map.npz \
  --model-path /path/to/model --endpoint http://localhost:30000 \
  --condition distance --out runs/external_map_interface/path32_distance.json
```

已完成的 128 维训练摘要见 [Step 1 报告](../cml_map_scaling/results/report.md)。原始 Q/V 记录在开发机的 `runs/cml_step1_explore_d813282/local128/...`；复制或重放时应在运行记录中标出来源。完整 Q 向量接口只需从同一 loader 序列化 Q，128 维接口可直接读取已训练的短地图；两者目前都不是首轮模型输入条件。不可将 1000 维 Q 直接截短并声称是训练后的 128 维地图。

## 当前积木训练：状态条件位移

`src/blocks_dynamics.py` 实现 `δ=MLP([Q(o),E(a)])` 与 `Q̂(o′)=Q(o)+δ`。Q 仍按单张初始棋盘中采样到的完整状态建表；动作 embedding 使用原有 `shape_id,row,col` 编号，一个共享 MLP 预测位移，动作空间不增加。候选距离为 `||Q(o)+δ−Q(goal)||₂`，不查真实后继 Q 来代替预测。

`src/train_blocks_dynamics.py` 复用下面旧试验的环境和转移采样，使用 [blocks_dynamics_pilot.json](configs/blocks_dynamics_pilot.json) 训练。采样包括构造解和随机合法 rollout，保留通向死局的合法动作；没有完整枚举状态图，也没有按好坏过滤转移。用户已确认首轮只训练转移 MSE，**不加入死局标签、路径距离或排序监督**。Q 的两个端点与 MLP 联合反向传播；这不同于旧图 `local_update` 只更新目标 Q 与动作 V 的规则，因此两者也不是只替换 V 参数化的严格消融。

训练保留连通 Q 状态的生成树和每个已出现动作，从剩余边留出一部分评测。这是已观察状态间的留出转移，不是未见棋盘泛化。`src/blocks_diagnostics.py` 分别测留出转移误差、相对不移动预测的误差、Q 尺度、从初始 Q 连续预测的逐深度偏移、候选预测距离以及死局排序。连续预测中不重置到真实 Q；缺少表内 Q 的实际后继只能统计覆盖缺口，不能声称其预测已被验证。立即死局与已有到目标路径的标签仅用于诊断。

```bash
python -m experiments.external_map_interface.src.train_blocks_dynamics \
  --config experiments/external_map_interface/configs/blocks_dynamics_pilot.json \
  --suite experiments/sol_dag_blocks/runs/suite.json --case-id blocks8_00 \
  --device cuda:0 --out runs/external_map_interface/blocks8_00_dynamics
```

产物目录保存模型、采样转移和固定划分、训练进度与诊断摘要，均在 Git 之外。此阶段训练积木 Q-map，不调用 LLM。实现需要 PyTorch；测试入口为 `python -m unittest discover -s experiments/external_map_interface/tests`。

## Previous baseline：积木共享动作 V

`src/blocks_q_map.py` 是**单张初始棋盘绑定**的最小训练试验。它直接复用 [BlocksTask](../sol_dag_blocks/src/tasks.py) 的 10×10 棋盘、`shape_id,row,col` 动作及合法移除，使用现有 case 的构造解和随机合法 rollout 收集 `(o1,a,o2)`。状态 Q 按这张棋盘内观察到的完整 mask 建表；同一个动作三元组在多个转移中共用一个 V 行。训练直接调用图 Step 1 的 `local_update`，没有改 Q/V 目标。产物保存在 `runs/`，不保证覆盖未见棋盘或所有可达状态。

`configs/blocks_q_200_rollouts_100_epochs.json` 与 `configs/blocks_q_expanded_pilot.json` 在同一 `blocks8_00` 棋盘上各训练 100 轮，分别采 200 与 2,000 次合法 rollout，用来检查更多 transition 对拟合误差与目标距离排序的影响。这些历史配置保留共享 `V_a` 定义，仍不是跨初始棋盘泛化实验。

```bash
python -m experiments.external_map_interface.src.blocks_q_map \
  --config experiments/external_map_interface/configs/blocks_q_pilot.json \
  --suite experiments/sol_dag_blocks/runs/suite.json --case-id blocks8_00 \
  --out runs/external_map_interface/blocks8_00_q_pilot
```

旧试验诊断只把无合法动作且非空的状态确认为 dead，把构造解上的状态确认为可解。报告比较“更少格子的 dead”与“更多格子的可解状态”的 learned Q 到目标距离，并统计已观察状态的合法后继有多少仍在 Q 表中。训练残差小并不推出这两类状态已经分离；初始棋盘绑定也不保证随机 rollout 覆盖全部后继。执行到未见棋盘后，表格无法直接取得 `Q当前`。更深层死局和跨棋盘泛化需后续单独检验。本试验不调用 LLM，也不把构造解或可解性标签交给 planner。

已完成的训练数值与局限见 [积木 Q/V 试验结果](results/blocks_q_pilot.md)。
