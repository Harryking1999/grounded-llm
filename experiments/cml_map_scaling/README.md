# 图地图与 Step 2 状态接口

本实验先从固定图轨迹学习 Q/V，再把当前节点和目标节点的 Q 各映射成一个连续 token，接入冻结的 Qwen。正式条件保存在 `configs/`，已审阅的摘要、判分记录和[简要结论](results/step2_acceptance.md)保存在 `results/`。原始输出、地图权重和模型 checkpoint 保存在忽略的运行目录。

代码按功能划分：`src/core.py`、`run.py`、`explore.py` 构建地图；`grounded_llm/graph_data.py`、`graph_prompts.py`、`graph_scoring.py` 统一负责题集、文字图和外部裁判；`src/step2_model.py` 负责 Adapter 与连续 token；其余 `step2*.py` 运行报告训练、完整文字对照、动作续训及重编号诊断。`tests/` 覆盖划分、评分、损失和各协议的关键边界。

先运行 `python -m unittest discover -s experiments/cml_map_scaling/tests -v`。完整训练与推理还需要配置指定的 Step 1 地图、Qwen 权重及 PyTorch 依赖。
