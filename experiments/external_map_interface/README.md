# 显式 cognitive map 接口

## 当前问题与边界

本目录是新的独立实验入口。图任务先复用 Step 1 的 Q/V 地图，逐步向 LLM 提供**候选动作的 learned-map 距离**。没有把 Q 压成 token，也没有训练 LLM 解码 Q。图与积木旧 Step 2 的代码、配置和历史结果留在原目录。

图的候选分数为 `||Q[current] + V[action] - Q[goal]||₂`。`V[action]` 是 learned-map 中动作的位移，不是动作本身。每次真实执行之后，从环境获取实际新节点，重新构造当步 prompt。图的动作目录按有向边定义，本身包含目标节点；因此当前图接口把真实候选终点作为动作语义给所有条件，learned-map 条件只额外提供距离。真实图最短路只在评测端使用，绝不进入 prompt 或 Q-map 读入模块。

`src/q_map.py` 只加载 Q/V 与计算候选距离；`src/transitions.py` 管合法动作和实际状态更新；`src/interface.py` 构造每步上下文；`src/planner.py` 调用模型与解析选择；`src/evaluate.py` 执行逐步评测。每步重新发送当前图和当步候选，避免把历史向量追加到上下文。图接口只露出标量距离，不露出完整或短 Q 向量。

## 图实验入口

`configs/path32_smoke.json` 是一个双起终点 smoke 合同。输入为 Step 1 同一 case 目录下的 `inputs.npz` 与 `map.npz`；可使用已训练的 128 维 Q/V，不需要额外 adapter。模型路径与运行 backend 是运行参数；有 SGLang 服务时传 `--endpoint`，在节点上直接加载模型时用 `--backend transformers`。产物写入 Git 忽略的 `runs/`。`plain`、`reasoning` 不读 Q/V；`distance` 在同一图和候选动作上增加 learned-map 距离。每步原始回答都会留档；非法动作、输出截断和步数上限分别计数。三条件共用生成预算；256 token 的初试中两个 thinking 条件都在首步截断，所以当前 smoke 使用 1024 token。三方正式比较仍需选定匹配的模型版本、提示、预算与测试起终点，当前 smoke 不能替代该比较。此前 Qwen Instruct 与 Qwen thinking 的历史结果配置不同，只能作参考。

```bash
python -m experiments.external_map_interface.src.evaluate \
  --config experiments/external_map_interface/configs/path32_smoke.json \
  --inputs runs/cml_step1_exploration/local128/official32/inputs.npz \
  --map runs/cml_step1_exploration/local128/official32/map.npz \
  --model-path /path/to/model --endpoint http://localhost:30000 \
  --condition distance --out runs/external_map_interface/path32_distance.json
```

已完成的 128 维训练摘要见 [Step 1 报告](../cml_map_scaling/results/report.md)。原始 Q/V 记录在开发机的 `runs/cml_step1_explore_d813282/local128/...`；复制或重放时应在运行记录中标出来源。完整 Q 向量接口只需从同一 loader 序列化 Q，128 维接口可直接读取已训练的短地图；两者目前都不是首轮模型输入条件。不可将 1000 维 Q 直接截短并声称是训练后的 128 维地图。

## 积木 Q/V 试验

`src/blocks_q_map.py` 是**单张初始棋盘绑定**的最小训练试验。它直接复用 [BlocksTask](../sol_dag_blocks/src/tasks.py) 的 10×10 棋盘、`shape_id,row,col` 动作及合法移除，使用现有 case 的构造解和随机合法 rollout 收集 `(o1,a,o2)`。状态 Q 按这张棋盘内观察到的完整 mask 建表；同一个动作三元组在多个转移中共用一个 V 行。训练直接调用图 Step 1 的 `local_update`，没有改 Q/V 目标。产物保存在 `runs/`，不保证覆盖未见棋盘或所有可达状态。

`configs/blocks_q_200_rollouts_100_epochs.json` 与 `configs/blocks_q_expanded_pilot.json` 在同一 `blocks8_00` 棋盘上各训练 100 轮，分别采 200 与 2,000 次合法 rollout，用来检查更多 transition 对拟合误差与目标距离排序的影响。它们沿用上述共享 `V_a` 定义，不引入 `V(Q,a)`；仍不是跨初始棋盘泛化实验。

```bash
python -m experiments.external_map_interface.src.blocks_q_map \
  --config experiments/external_map_interface/configs/blocks_q_pilot.json \
  --suite experiments/sol_dag_blocks/runs/suite.json --case-id blocks8_00 \
  --out runs/external_map_interface/blocks8_00_q_pilot
```

诊断只把无合法动作且非空的状态确认为 dead，把构造解上的状态确认为可解。报告比较“更少格子的 dead”与“更多格子的可解状态”的 learned Q 到目标距离，并统计已观察状态的合法后继有多少仍在 Q 表中。训练残差小并不推出这两类状态已经分离；初始棋盘绑定也不保证随机 rollout 覆盖全部后继。执行到未见棋盘后，当前表格无法直接取得 `Q当前`；这一步需先解决，才能将积木地图接入逐步 planner。更深层死局和跨棋盘泛化需后续单独检验。本试验不调用 LLM，也不把构造解或可解性标签交给 planner。
