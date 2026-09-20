# 实验 harness

## 阅读入口

| 研究 | 内容 |
| --- | --- |
| [cml_map_scaling](cml_map_scaling/README.md) | 当前主线：Q/V 地图结果与 Step 2 接口 runbook |
| [qwen_path_blocks](qwen_path_blocks/README.md) | Qwen thinking 寻路／积木基线与 Sol 小批参考 |
| [sol_dag_blocks](sol_dag_blocks/README.md) | Sol 新形状积木与有向图基线 |
| [gcml_counterexamples](gcml_counterexamples/README.md) | Luna／Flash 基线与错误案例 |
| [state_interface_pilot](state_interface_pilot/README.md) | 后续积木接口设计，尚未运行 |

## 目录约定

只有在具备具体假设或评测器时，才创建研究目录。默认结构为：

```text
experiments/<study>/
  README.md          假设、baseline、指标、当前结论
  configs/           已提交的机器可读正式运行合同
  src/               可复用的环境或方法实现
  tests/             针对性的正确性测试
```

生成状态应放在受跟踪源码之外：

```text
runs/<run_id>/       解析后的配置、预测、指标、日志、产物
```

Git 会忽略 `runs/`。只有紧凑且与决策有关的结果，才应提升到研究 README 或权威项目状态中。

## 最小运行合同

正式运行应明确：

- 模型与推理设置；
- 环境／规则版本与数据集划分；
- 观测编码和可用工具；
- 适用时的随机种子；
- 主要指标；
- 源代码 commit 与输出目录。

探索性 smoke 不需要生产级 manifest。只记录足以复现那项会影响下一步决策的观察即可。
