# 地图读取接口：首阶段整体执行方案

本文件按[飞书方案](https://zcnhpsd26tyi.feishu.cn/wiki/ZNNSwpIo0iHwMgk7mk7coBA3nzb)的章节方式，整理我们已确认的寻路实验。机器可读参数以[正式配置](configs/pilot_path256.json)为准。

## 一、目标与本阶段范围

用已有 Q/V 地图提供状态与动作后果，让冻结的 Qwen2.5-1.5B-Instruct 通过可训练的 cross-attention 读取地图，生成距离比较、动作选择与终止总结。首阶段先做寻路。旧文字地图实验沿用原协议；同一地图上的贪心选择作为参照。

## 二、任务与数据

使用五张 256 节点图及各自已有的 128 维 Q/V。四张图用于训练和验证，第五张图留作测试。原有题集的 120 个起终点对全部保留为测试题，其反方向也不进入训练或验证。

每张训练图按环境最短步数 4、6、8、10 分层；每档采 64 条训练和 16 条验证轨迹，再采 32／8 条起点即目标的训练／验证轨迹。四图合计 **1,152 条训练、288 条验证**。每条样本是完整的成功交互轨迹，最多 32 次实际动作。每轮保留所有合法且未访问的邻居，随机重排候选编号。示范动作选预测后继中地图距离最小者；并列时随机选一个。遇到死路或超过动作上限的采样记录原因并补采。

分层采样覆盖不同路径长度；增加题目数及每次更新混合多张图，是为让梯度同时反映不同起终点。候选乱序使编号只表示本轮候选，不能固定指向某个动作。

数据清单已经生成。因超过动作上限，四图分别排除 1、0、4、2 条采样；死路排除数均为 0。用目标模型 tokenizer 编码后，训练完整轨迹中位数为 6,814 token、最大为 14,501 token；验证最大为 13,774 token。两组均处于确定的 16,384 token 输入上限内。

## 三、完整交互流程

初始英文提示逐字沿用旧寻路实验的任务开头及完整邻接表，给出起点、目标和不重复访问规则。此后执行器每轮追加英文 `[Environment update]`，其中包含实际当前节点、实际执行路径，以及本轮随机编号的合法候选 `i: current -> destination`。旁路同时提供本轮地图 \(M_t\)。

初始提示保持一致，是为了沿用原寻路任务与图信息；逐轮更新提供真实动作反馈，使文字上下文与当前环境同步。

LLM 输出距离、排序、选择依据和完整的 `<action>i</action>`。执行器按**本轮**编号映射校验并执行动作，取得真实后继状态，追加下一条环境更新，再继续生成。到达目标后仍追加一轮，让 LLM 总结实际动作并输出 `<done/>`；起点即目标时直接进入这一轮。`<done/>` 由环境核验，之后不执行动作。

这是一条持续增长的对话轨迹：遇到动作标记暂停，执行并更新环境，再在原有上下文上续写。**训练时整条轨迹一次前向、反向，不按轮切成独立样本。**

完整训练保留早期行动和后期总结之间的上下文关系，也让各轮地图按照文字位置分别绑定。

| 时刻 | LLM 的文字上下文新增内容 | 旁路地图 | 后续 |
| --- | --- | --- | --- |
| 初始 | 邻接表、起点、目标、规则 | — | 进入第 0 轮 |
| 第 \(t\) 轮 | 当前节点、实际路径、本轮编号的合法候选 | \(M_t\)：当前、目标、各候选预测后继 | 生成 `<action>i</action>` 或 `<done/>` |
| 动作后 | 执行器追加新的环境更新 | 用真实状态重建 \(M_{t+1}\) | 继续同一轨迹 |

候选编号只在当轮有效。实际动作日志使用 `move(source,destination)`，而非历史候选编号。每轮文字不预填地图距离、排序或已选动作。

## 四、架构与训练

### 4.1 架构

#### 1）模块分工

环境／执行器维护真实节点、已访问路径、合法动作和终止核验；冻结的 Q/V 提供地图状态与动作后果；新增读取接口把地图记忆送入冻结的语言模型；LLM 负责文字判断与候选编号输出。执行器只接受本轮存在且合法的编号，执行后按实际状态更新。

#### 2）地图输入：当前、目标、候选后继

令 \(o_t\) 为实际当前状态，\(o^*\) 为目标，\(a_i\) 为第 \(i\) 个合法动作：

\[
z_t=Q(o_t),\qquad z^*=Q(o^*),\qquad
\hat z_{t+1}^{(i)}=z_t+V(a_i,o_t).
\]

本轮地图按 `[current, goal, successor_1, …, successor_k]` 排列。首版直接输入预测后继向量，提前完成 \(z_t+V\) 的加法，供接口比较后继与目标；不额外输入精确距离标量。真正执行后，下一轮使用环境返回的 \(o_{t+1}\) 重新计算 \(Q(o_{t+1})\)；目标不变时保留 \(z^*\)。

#### 3）角色与编号编码

每个状态向量经共享投影 \(P\)，与角色、编号嵌入拼接，再由 \(G\) 映射到语言模型 hidden size。例如候选 \(i\)：

\[
m_i=G\big([P(\hat z_{t+1}^{(i)});E_{\rm role}(\mathrm{successor});E_{\rm id}(i)]\big).
\]

当前和目标同样编码，分别使用 `current`、`goal` 角色及 `none` 编号。地图记忆 \(M_t=[m_{\rm current},m_{\rm goal},m_1,\ldots,m_k]\)。候选数可变，批处理时补齐槽由 attention mask 屏蔽。

首版把 128 维地图向量投影到 256 维，角色与编号各用 32 维，再组合到 Qwen 的 1,536 维 hidden size。角色区分向量用途，编号负责连接候选槽与文字中的 `<action>i</action>`。

#### 4）通过 cross-attention 读取

在 Qwen 的每个解码层**之前**插入门控地图 cross-attention，共 28 层。文字 hidden states \(H_\ell\) 形成 query，本轮地图 \(M_t\) 形成 key/value。每层 4 头、每头 64 维；省略多头拼接后的具体 reshape，核心计算是：

\[
Q_\ell=H_\ell W_\ell^Q,\quad K_\ell=M_t W_\ell^K,\quad
V_\ell=M_t W_\ell^V,
\]
\[
H'_\ell=H_\ell+\tanh(g_\ell)
\operatorname{CrossAttn}_\ell(Q_\ell,K_\ell,V_\ell),\qquad g_\ell=0\ \text{at initialization}.
\]

原 Qwen 参数及 Q/V 地图冻结。共享 \(P/G\)、角色／编号嵌入、各层 cross-attention 和门控在**同一次 SFT** 中联合更新。架构借鉴 Flamingo 的冻结语言模型与门控接入；本实验直接接入地图向量。

冻结已有模块后，训练信号集中更新地图到语言的读取接口；零门控让新增层初始不改变原模型的输出。4 头是首版在接口容量与显存开销之间选定的配置。

#### 5）文字位置与地图绑定

完整轨迹保存 \(M_0,M_1,\ldots\)，每个文字 token 附所属轮次。cross-attention mask 令该 token 只读对应的 \(M_t\)。第一轮回答直到完整 `</action>` 都绑定 \(M_0\)；下一条环境更新及回答绑定 \(M_1\)。普通因果 self-attention 仍能读取此前文字。绑定由旁路 mask 实现，提示中不加 `<map>`。推理在环境更新后重新计算累积前缀。

#### 6）一条完整的数据例子

用三节点图 `0—1—2` 示意；真实训练使用 256 节点图。设起点 1、目标 2，\(Q(0)=(0,0)\)、\(Q(1)=(1,0)\)、\(Q(2)=(2,0)\)，V 准确预测沿边移动。第一轮候选随机编号为 `1: 1 -> 0`、`2: 1 -> 2`。

```text
USER
Find a valid path from node 1 to node 2 in this undirected graph, as short as you can.
Use the listed edges and visit each node at most once.

Neighbors:
0: 1
1: 0, 2
2: 1

[Environment update]
Current node: 1
Actual executed path: [1]
Legal next moves:
1: 1 -> 0
2: 1 -> 2
Candidate numbers may change between turns; use only the numbers above.
If the current node is the goal, summarize the actual executed path and end with <done/>.
Otherwise, report the current-to-goal distance and each candidate successor's distance to the goal, rank candidates from closest to farthest, explain your choice, and end with <action>i</action>.
[/Environment update]
```

旁路 \(M_0=[Q(1),Q(2),Q(0),Q(2)]\)，槽角色为 `[current, goal, successor, successor]`，编号为 `[none, none, 1, 2]`。本轮目标回答为：

```text
ASSISTANT
The current node has not reached the goal.
Current-to-goal map distance: 1.0000.
Candidate successor-to-goal map distances: 1=2.0000; 2=0.0000.
Ranking from closest to farthest: 2, 1.
Candidate 2 is closer to the goal than the current node; candidate 1 is farther.
Choose candidate 2 because its predicted successor has the smallest map distance.
<action>2</action>
```

执行器据本轮映射执行 `move(1,2)`；随后实际状态和地图刷新为当前节点 2、\(M_1=[Q(2),Q(2)]\)：

```text
USER
[Environment update]
Current node: 2
Actual executed path: [1, 2]
Legal next moves:
(none)
Candidate numbers may change between turns; use only the numbers above.
If the current node is the goal, summarize the actual executed path and end with <done/>.
Otherwise, report the current-to-goal distance and each candidate successor's distance to the goal, rank candidates from closest to farthest, explain your choice, and end with <action>i</action>.
[/Environment update]

ASSISTANT
The current node is the goal.
Executed actions: move(1,2).
Summary: Reached node 2 after one executed move.
<done/>
```

### 4.2 训练数据与训练方法

#### 1）监督内容

动作段监督“尚未到目标 → 当前和候选的地图距离 → 排序及比较 → 选择依据 → `<action>i</action>`”；终止段监督“已到目标 → 实际动作序列及简短总结 → `<done/>`”。地图距离为

\[
d_{\rm map}(z,z^*)=\|z-z^*\|_2.
\]

数字显示四位小数；排序和并列按未舍入数值确定。初始提示及环境更新作为上下文，只有 assistant 回答 token 参与交叉熵损失。

#### 2）优化与参数更新

每次优化器更新累计 **8 条完整轨迹**的梯度，一次处理一条，尽量从四张训练图各取两条。设 \(A_j\) 是第 \(j\) 条轨迹中全部 assistant 回答 token，则

\[
\mathcal L=\frac{\sum_{j=1}^{8}\sum_{p\in A_j}\operatorname{CE}_{j,p}}
{\sum_{j=1}^{8}|A_j|}.
\]

即每个受监督 token 等权。最多训练两个 epoch；AdamW、学习率预热、精度等细节见正式配置。

按已确认的首轮设置，AdamW 学习率为 \(2\times10^{-4}\)，前 5% 更新预热；冻结底座用 bfloat16，新增参数保持 float32，梯度范数裁剪至 1。每次积累八条完整题目，扩大一次更新所覆盖的样本与图。

#### 3）检查点与运行目录

每 25 次优化器更新或 15 分钟保存检查点，epoch 末也保存。检查点包含新增模块参数、优化器、调度器、数据位置和随机状态，可从最近的完整更新继续。远端独立目录 `/zhanghanyue/experiment/flamingo_map_reader/` 分设 `code/repo/`、`runs/`、`logs/`、`models/`、`results/`、`reports/`。

#### 4）评测

分别报告地图距离误差、候选排序、动作选择、闭环到达、最短路径和终止错误；同时记录非法编号、预算耗尽与生成成本。分析条件包括候选重排、目标地图替换和地图遮蔽。评测时模型自由生成，执行器按实际动作刷新状态，环境核验结果。

## 五、参考工作与对应设计

| 依据 | 本方案采用的部分 |
| --- | --- |
| [飞书方案 4.1／4.2](https://zcnhpsd26tyi.feishu.cn/wiki/ZNNSwpIo0iHwMgk7mk7coBA3nzb) | 当前／目标／候选后继的地图输入、角色和编号、逐轮输出及 SFT 内容 |
| [Flamingo 论文](https://arxiv.org/abs/2204.14198) | 冻结语言模型，使用门控 cross-attention 读取外部记忆 |
| [OpenFlamingo 的注意力实现](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py)与[层包装](https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/flamingo_lm.py) | 层前插入读取模块，按文字位置屏蔽不属于该位置的外部记忆 |
| [ReAct 论文](https://arxiv.org/abs/2210.03629) | 推理、行动与环境反馈交替进入同一轨迹 |
| [Qwen2.5-1.5B-Instruct 模型配置](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) | 28 个解码层的具体插入位置 |

## 六、当前进度

已完成读取接口骨架、英文提示与目标回答模板、完整轨迹的 token 到地图绑定、数据清单和 tokenizer 长度检查。接续工作是训练入口、单条完整轨迹的显存／耗时检查、正式训练及逐项评测。
