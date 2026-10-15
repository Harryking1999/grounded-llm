# 二阶魔方 Qmap

**当前保留六动作纯 Q 基线，长程求解尚未成功。**整体模型先看[共享说明](../puzzle_qmap/README.md)，实测结论看[结果报告](results/report.md)。这里只解释任务和数据。

## 状态与动作

用八个角位置的角块身份 cp 与朝向 co 表示状态。角块身份为 0…7 的排列，朝向为 0/1/2，朝向和必须模 3 为零。采用 [Kociemba 角块约定](https://www.kociemba.org/math/cubielevel.htm)，不施加三阶魔方的角块／棱块奇偶耦合限制。

使用 DBL 角块固定且不扭转的规范参考系，共 7! × 3⁶ = **3,674,160** 个状态。接口要求输入已经处于此参考系，不包含贴纸识别或自动规范化。网络输入是“位置 × 角块身份 × 朝向”的联合独热，共 192 维；状态编号只是索引，不进入网络。

当前任务名 cube2_qtm，动作是 U、U′、R、R′、F、F′；从相应面外侧看，正转为顺时针 90°，每次动作计一步。每个状态均有六个合法后继；180° 需两步。图直径为 14，每条边到固定目标的真距离变化都为 ±1。环境仍可读取历史九动作 HTM 约定，旧权重与六动作权重不能混用。

## 训练数据

1. 从复原态完整 BFS。任意两状态的距离通过群相对状态查表，不能拿打乱序列长度作标签。
2. 每个相对根状态枚举全部六个单步和 36 个两步序列，合并为 34 个不同状态。起点及六个直接后继作为七个父节点，各自保留完整六候选。
3. 按根状态到目标的真距离分层，每层打乱、无放回遍历；抽到后将目标与整个局部图共同左乘同一个 k。此变换保持距离和动作边；对每个状态执行同一串右乘动作不能替代它。
4. 距离对包含节点到目标的长程关系和局部节点间的近程关系，均使用完整图最短距离。排序也使用全部真实候选，保留所有最优动作。
5. 训练图中所有输入排除保护的评测端点。data.npz 中的训练数组只是固定监测集，实际训练在线构造新呈现；根层均衡不等于展开后的所有距离层都严格均衡。

## 配置与运行

只保留一个基线配置：[quarter_turn_local.json](configs/quarter_turn_local.json)。清理后的默认配置可独立生成划分，无需旧运行目录；它不是已报告历史运行的原样复现，旧配置和数据在归档中。可选 reference_data 仅用于需要保持历史同题比较的情况。

```bash
python -m experiments.puzzle_qmap.src.run --config experiments/cube2_qmap/configs/quarter_turn_local.json --out runs/cube2_qmap/RUN_ID --source-commit COMMIT --device cuda:0
python -m experiments.puzzle_qmap.src.solve --task cube2_qtm --start 12345 --goal 0 --checkpoint runs/cube2_qmap/RUN_ID/best.pt
python -m unittest discover -s experiments/cube2_qmap/tests -v
```

运行目录必须不存在。纯 Q 求解不加载 oracle；精确求解是独立裁判条件。
