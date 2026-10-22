# Token Swapping公开实例与适配

可直接读取的公开实例位于[Quantinuum/tket固定问题](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/Data/FixedCompleteSolutions.cpp)，其[说明](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/Data/FixedCompleteSolutions.hpp)明确保留架构的全部边，并提供现有求解器给出的交换序列；**不保证参考序列最短**。这是软件固定测试实例集，不能称为统一LLM benchmark或已证明最优的标准题库。数据按上游Apache-2.0许可提供，原始文件保留版权声明。

另有[Symbolic Algorithms for Token Swapping](https://si2.epfl.ch/~demichel/publications/archive/2020/ISMVL_bruno.pdf)使用量子映射中的8338个实例，区分最少顺序交换和并行轮数；[OOPSLA2019公开artifact](https://zenodo.org/records/3366904)涉及量子分配和Token Swapping。本项目尚未核实这两处能否直接提取带精确最优标签的独立实例；本次不冒充复现它们的题集或成绩。

[准备入口](../../../experiments/task_text_baselines/src/prepare_rush_token.py)读取固定提交，遵循[上游解码](https://github.com/Quantinuum/tket/blob/4a554cbc2699de2f95da18eb3c30ff0961bcc536/libs/tktokenswap/test/src/TestUtils/DecodedProblemData.cpp)：架构字符串含各顶点较大编号的邻接点；交换序列每两个字符为一条边，下划线后是初始有棋子的顶点。仅采用所有顶点都有棋子的完整映射，并按源解码约定取交换产生排列的逆排列作为初始“各顶点上的目标编号棋子”，原始正向交换序列应使其归位。保留全部架构边，不能仅用参考解经过的边重新建图。

精确标签通过有界双向BFS重新求得，逐层完整扩展；参考解还需逐步回放。按[配置](../../../experiments/task_text_baselines/configs/deepseek_rush_token64k.json)的八种架构等额抽取，排除同架构重复排列。若超出真值计算界限不伪造最优标签，保留拒绝数；实际冻结题目和原始序列保存在`runs/`。模型输入仅图、初始排列、目标和规则；参考解、最优数与搜索不进入提示。

判分记录所有提交交换，包含首次到达目标后又执行的动作。完成答案另报`cost - optimum`和`cost / optimum`，失败答案不与成功答案混算平均成本，最优成功数仍以全部有效题为分母。

[example.json](example.json)仅说明格式。

## 保留的Qmap场景

第一版固定[tree10.json](tree10.json)所定义的公开10顶点树，学习任意完整排列之间的距离；与已测`Medium random tree 1`的5个Flash实例使用同一底图。这里固定的是环境，不把这5道题当作地图全部数据；12–15顶点留待首版验收后单独准备。

完整归位距离表与状态对训练数据是不同产物：前者是环境真值，后者控制Qmap实际曝光；可通过相对排列查询任意状态对的精确距离。具体编码、留出、验收及未实施部分见[Qmap说明](../Qmap/README.md)。原Flash归位题与新增任意目标题不自动构成同题比较，新增条件必须重新匹配文字与地图输入。
