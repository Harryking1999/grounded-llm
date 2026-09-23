# 图地图与 Step 2 状态接口

本实验先从固定图轨迹学习 Q/V，再把当前节点和目标节点的 Q 各映射成一个连续 token，接入冻结的 Qwen。`configs/` 保存运行配置，`results/` 保留一份关键数值摘要和[简要结论](results/step2_acceptance.md)。逐题输出、判分留档、地图权重和模型 checkpoint 放在忽略的运行目录。

代码按功能划分：`src/core.py`、`run.py`、`explore.py` 构建地图；`grounded_llm/graph_data.py`、`graph_prompts.py`、`graph_scoring.py` 统一负责题集、文字图和外部裁判；`src/step2_model.py` 负责 Adapter 与连续 token；其余 `step2*.py` 运行报告训练、完整文字对照、动作续训及重编号诊断。`tests/` 覆盖划分、评分、损失和各协议的关键边界。

训练与推理直接读取已有的 Step 1 地图和 Qwen 权重，需要 PyTorch 等依赖。修改数据、评分或损失后，运行相应模块的测试；小批训练检查可用 `--mode smoke`。
