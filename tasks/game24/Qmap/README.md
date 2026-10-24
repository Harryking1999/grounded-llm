# 24点 Qmap

编码是四个排序槽位，各含存在掩码、带符号log(1+|分子|)、log(1+分母)、有界数值；共享 MLP 与 directed_max。Q只读当前值，不读解或精确可解标签。到[24]的可解状态距离恒为剩余数字个数减1，因此这轮主要检验可解性和动作选择。

训练复用 [StateEncoder、distance、soft_rank_loss](../../../grounded_llm/core/qmap/model.py)，保持距离Huber＋原权重软排序；按任务独立运行[首轮实验](../../../experiments/new_tasks_qmap/README.md)。不加入任务专属学习损失。不可达以配置中有限代理值训练与排序；它不是精确无穷距离。评测始终保留合法死路，并报告成本、动作顺序和完整求解。

尚无采用权重，selection保留空值。首轮候选保存在运行目录。
