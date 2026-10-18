# 汉诺塔任务工作区

当前采用八盘Qmap，适用范围为**任意合法起点到指定柱上的整塔终点**。全状态整塔到达99.89%，六个整塔方向6/6；仍保留21个失败起点及两个循环。中间目标结果作为补充，不作为本轮采用门槛。来源与范围见[选用记录](Qmap/selection.json)，三至六盘权重仅作历史参考。

- [文字、规则与各组提示](prompts/README.md)：任务内原文；根目录 `TASK_TEXTS.md` 是自动集中副本。
- [数据准备与示例](data/README.md)：小例说明字段，不冒充正式训练集。
- [Qmap](Qmap/README.md)：训练、加载、选定权重与适用范围。
- [环境规则](../../experiments/hanoi_qmap/src/environment.py)；[研究协议与实现](../../experiments/hanoi_qmap/README.md)。
- 当前进展与下一步只维护在 [项目状态页](../../docs/PROJECT_STATUS_AND_TODO.md)。

按本任务独立准备数据、训练和评测。共用 MLP、距离与排序损失等基础组件；不同任务的状态编码、动作、采样与训练入口分别维护。

```text
hanoi/
  README.md
  prompts/README.md
  data/README.md
  data/example.json              # 三盘字段说明例
  Qmap/
    README.md                   # 八盘地图加载、训练与评测入口
    selection.json              # 当前八盘选用范围；历史六盘引用单列
    checkpoints/                # Git忽略；本地实际权重
      hanoi8_full_coverage_fb7ffe6_best.pt
      hanoi6_best.pt             # 历史六盘参考
```
