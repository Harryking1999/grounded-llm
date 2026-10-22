# Token Swapping（图上交换棋子）

采用[顺序相邻交换问题](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ESA.2016.66)：每个顶点恰有一个不同棋子，一次交换一条无向边两端的棋子，目标全部归位，成本为总交换次数。完整计划已接入，尚无地图或LLM训练。

Flash high、64k在公开40题上完成36/40，全部成功计划最优；4份失败均在10顶点树上耗尽思考预算、最终答案为空。40题均由本地精确搜索重新标定最优成本；4条源序列本身不是最短，不能直接作最优标签。完整结果见下方统一结果。

- [规则与文字](prompts/README.md)
- [公开实例、适配与真值](data/README.md)
- [地图条件](Qmap/README.md)、[采用记录](Qmap/selection.json)
- [规则实现与精确求解](../../grounded_llm/tasks/token_swapping/environment.py)
- [Flash筛查合同](../../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)、[统一结果](../../experiments/task_text_baselines/results/report.md)

连通底图上所有完整排列均可达；NP-hard目标是最少交换数，不是任意完成。评测须保留完成、最优率和超额交换数，不能只比较完成率。
