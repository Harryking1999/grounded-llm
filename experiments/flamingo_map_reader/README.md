# 地图读取接口

假设：冻结语言模型，加入 Flamingo 风格的门控 cross-attention 后，模型能够从 Q/V 地图记忆中读取候选与目标的关系并输出候选编号。对照为相同语言模型的无地图输入，以及相同地图上的贪心选择；距离读取、动作选择和闭环到达分别评估。

首阶段使用寻路任务和已有 Q/V，按实际合法动作构造预测后继；环境的真实状态负责执行与终止核验。已实现地图记忆编码、门控接入、候选编号映射及可恢复的 adapter／优化器检查点。地图组与无地图组共用旧寻路实验的初始任务提示和邻接表，逐轮提供实际路径与随机编号的未访问邻居；两组使用相同的英文逐轮指令。首版冻结 Qwen2.5-1.5B-Instruct。监督文本、正式配置和训练尚待确定。

代码在 `src/`，针对性检查在 `tests/`。正式运行配置放 `configs/`；运行、检查点和原始日志放 Git 忽略的 `runs/` 与 `logs/`，紧凑结果放 `results/`，可读报告独立保存。

接入方式参考 [Flamingo](https://arxiv.org/abs/2204.14198) 和 [OpenFlamingo 的层包装实现](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/flamingo_lm.py)：在冻结的解码器层前加入零初始化门控的 cross-attention。本实验直接接入少量地图向量，不使用视觉编码器、Perceiver 或 `<image>` 标记。
