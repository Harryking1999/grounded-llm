# 功能唯一实现

本分支复用 `codex/experiment-harness` 的已提交实现（`cdd9e58`），通过
`python -m grounded_llm.harness --config ...` 运行；不另写 Blocks 训练器。

| 功能 | 实现 |
| --- | --- |
| 配置继承与任务分派 | `grounded_llm/config.py`、`harness.py` |
| 冻结模型与 tokenizer | `grounded_llm/model.py` |
| Linear／MLP 和向量槽替换 | `grounded_llm/interface.py` |
| 答案 loss、优化、训练、选模、权重保存 | `grounded_llm/training.py` |
| 模型 generation 调用 | `grounded_llm/inference.py` |
| JSON 产物读写 | `grounded_llm/artifacts.py` |
| 图数据、模板、裁判 | `grounded_llm/data.py`、`prompts.py`、`scoring.py` |
| 积木规则和真实转移 | 已有 `experiments/sol_dag_blocks/src/tasks.py::BlocksTask` |
| 积木数据、逐步严格判分、配对指标 | `grounded_llm/blocks.py` |
| 不同覆盖率、中间态、孤立格的读出数据 | `grounded_llm/blocks_readout.py` |
| 像素输入、状态报告序列、JSON 边界 | `grounded_llm/blocks_interface.py` |
| 积木训练回调和逐步 episode 编排 | `grounded_llm/blocks_run.py` |

积木接口仅补充图任务没有的原始像素输入和同一 assistant 序列的动作边界。
所有状态向量由共享 MLP 与槽替换接口处理；训练复用共享 loss 和训练循环。
模型不接触构造见证、合法动作列表或裁判搜索。配置引用旧规则，不复制形状表。
正式参数见 `experiments/state_interface_pilot/configs/blocks_step2.json`；smoke 只继承覆盖样本量。

公共模块的图任务兼容接口保留；本分支没有合并其他分支的图实验进展或重跑图实验。
