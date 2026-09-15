<h1 align="center">基于 Jittor 的时序图候选排序</h1>

<h3 align="center">赛道一 · A 榜第 7 名 · B 榜第 2 名</h3>

> 一条边发生之后，下一条边会走向哪里？

`jittor-sader-jituai` 保留了算法实现、训练与推理链路及结果重建工具。

<p align="center">
  <a href="https://github.com/Jittor/jittor"><img src="https://img.shields.io/badge/Framework-Jittor-0ea5e9?style=flat-square" alt="Jittor"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square" alt="Python 3.10"></a>
  <a href="#competition"><img src="https://img.shields.io/badge/Task-Temporal%20Graph%20Recommendation-8b5cf6?style=flat-square" alt="Temporal Graph Recommendation"></a>
</p>

<p align="center">
  <a href="#competition">赛题说明</a> ·
  <a href="#jittor">Jittor 优势</a> ·
  <a href="#a-list">A 榜具体方案</a> ·
  <a href="#b-list">B 榜具体方案</a> ·
  <a href="#reproduce">复现入口</a>
</p>

<p align="center">
  <a href="https://commons.wikimedia.org/wiki/File:Collaborative_filtering.gif">
    <img src="https://upload.wikimedia.org/wikipedia/commons/5/52/Collaborative_filtering.gif" alt="协同过滤中的用户与物品交互" width="480">
  </a>
</p>

<a id="competition"></a>

## 1. 赛题说明

赛道一研究**时序交互图上的下一目标预测**。训练数据由按时间发生的
`(src, dst, time)` 交互边组成；测试时，每条查询给出来源节点、必要的时间字段
以及固定的 100 个候选目标。模型不需要从全体节点召回，只需要回答：
**真实目标在这 100 个候选中应排第几。**

### 一个具体例子

假设测试集中有一条查询：

| 查询元素 | 示例                                                    |
| -------- | ------------------------------------------------------- |
| 来源节点 | 用户/节点`42`                                         |
| 查询时刻 | `10:30`                                               |
| 官方候选 | `[13, 7, 88, 21, 5, ...]`，共 100 个                  |
| 可见历史 | `42 -> 13` 发生在 5 分钟前；`42 -> 7` 发生在 3 天前 |

模型只能读取 `10:30` 以前的边。时序特征会认为候选 `13` 更近期，embedding
会判断 `42` 与各候选的长期兼容性，集合模型再比较这 100 个候选的相对关系。
例如模型输出：

```text
候选列: [13,   7,   88,  21,  5,   ...]
模型分: [0.82, 0.47, 0.09, 0.31, 0.18, ...]
排序结果: 13 > 7 > 21 > 5 > 88 > ...
```

若隐藏的真实目标是候选 `21`，它排第 3，本行倒数排名为
`RR=1/3`。测试时真实目标不可见；它只由评测端用于计算名次。

提交时仍按官方候选列顺序写回 100 个数，不能把候选重新排成另一组，也不能
加入候选集合以外的节点。

设第 `i` 条查询为来源 `s_i`、查询时刻 `t_i` 和**有序候选向量**
`C_i=(c_i1,...,c_i100)`，模型为每个候选产生分数：

```math
z_{i,j}=f(H_{\le t_i},s_i,c_{i,j},C_i),
\qquad j=1,\ldots,100.
```

训练历史严格限制在查询时刻以前；部分 transductive 结构规则还会读取整份
测试表中**无标签**的来源、时间和候选共现，但从不读取隐藏目标。提交文件第
`j` 列始终对应 `c_{i,j}`，重复候选和并列分数也按原列稳定处理。

对每条查询，`RR=1/rank`；数据集 MRR 是所有查询 RR 的平均值。仓库训练代码
按两个数据集的 MRR 汇总榜单分，因此记录总分可以大于 1。

| 榜单 | 官方数据       | 数据集   |            提交矩阵 | 输出形式       |
| ---- | -------------- | -------- | ------------------: | -------------- |
| A 榜 | `data_A.zip` | Dataset1 |    `61,051 x 100` | 候选概率       |
| A 榜 | `data_A.zip` | Dataset2 |   `153,420 x 100` | 候选概率       |
| B 榜 | `data_B.zip` | Dataset3 | `157,670 x 100` | 八位小数概率（行和为 1） |
| B 榜 | `data_B.zip` | Dataset4 | `2,322,538 x 100` | 构建器固定 rank grid |

记录成绩分别为 A 榜 `1.521072794155721`、B 榜
`1.5240999401892983`。

每个数据集目录包含训练与测试 CSV。训练表提供 `src/dst/time` 以及数据中已有
的 `split` 字段；测试表提供 `src/time/c1...c100`。最终 `result.zip`
包含两个无表头 CSV，行序和候选列序必须与官方测试表一致。

四个数据集共享同一候选排序骨架；具体专家是对各数据字段和规模的实例化：

```mermaid
flowchart LR
    A["官方历史 + 100 候选"] --> B["Jittor 图 / 时序专家"]
    B --> C["候选上下文 + 行内融合"]
    C --> D["结构校正 + Dataset1-4 输出"]
```

**不变的部分**

- 只使用官方数据和无标签候选结构，不读取测试标签，不引入外部数据。
- 所有神经网络训练与前向都使用 Jittor；最终推理只给官方 100 候选赋值。
- 各成员在数据集内部完成融合或结构校正，不改变候选身份和列位置。
- 输出固定候选列、固定序列化规则和 SHA-256 校验。

**只因数据而变化的部分**

- Dataset1/3 更偏向共享节点空间中的图邻域、来源历史和 session 支持。
- Dataset2/4 更偏向稀疏交互、时间切片、隐式反馈和候选集合关系。
- B 榜数据规模更大，因此增加种子数、历史窗口、流式 cache 和分块推理。
- A 榜与 B 榜 Dataset3 输出候选概率；B 榜 Dataset4 构建器写入固定 rank grid。

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <a href="https://commons.wikimedia.org/wiki/File:Barabasi_Albert_model.gif">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/48/Barabasi_Albert_model.gif" alt="Barabasi Albert 网络生长" width="100%">
      </a>
    </td>
    <td align="center" width="50%" valign="top">
      <a href="https://commons.wikimedia.org/wiki/File:Social_graph.gif">
        <img src="https://upload.wikimedia.org/wikipedia/commons/d/de/Social_graph.gif" alt="社交图逐步展开" width="100%">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center"><b>网络生长</b><br>新交互改变节点的局部结构与热度。</td>
    <td align="center"><b>关系展开</b><br>来源、目标和历史形成候选上下文。</td>
  </tr>
</table>

<a id="jittor"></a>

## 2. 为什么这道题用 Jittor

这是**计图（Jittor）人工智能挑战赛**的算法赛道。使用 Jittor 完成模型设计、
训练和预测不只是框架合规要求，也决定了项目如何组织多种排序成员。

验证基线固定为 **Ubuntu 22.04 + NVIDIA RTX 4090 + CUDA 12.4 +
Python 3.10 + Jittor 1.3.10.0**；入口会在训练或推理前检查版本、CUDA 和
Jittor 算术探针，环境不符时直接停止。

Jittor 官方将其核心概括为 **JIT 动态编译、元算子和统一计算图执行**：
Python 前端保留动态图式的开发体验，CUDA/C++ 后端负责编译和优化执行。
本项目没有虚构跨框架加速比；下面只说明这些机制在仓库里的实际价值。

| Jittor 机制 | 对本方案的价值 | 仓库中的落点 |
| --- | --- | --- |
| JIT 编译与算子优化 | 按实际模型和张量形状编译执行路径，适配 MLP、embedding、MF 与 attention 的不同计算图 | A/B 所有神经成员 |
| 元算子与 Python 前端 | 用统一的 `Module / Var / nn` 接口组合图特征、稀疏偏好、集合注意力和残差网络 | D1-D4 模型定义 |
| 统一计算图与自动求导 | 同一训练循环可覆盖交叉熵、BPR、VAE 与混合排序损失 | `AdamW`、`softplus`、`cross_entropy_loss` |
| CUDA 后端 | 大候选组、embedding 表和分块前向直接运行在 GPU；启动器显式检查 `jt.has_cuda` | A raw training、B C2/RUC4/`third_1` |
| 状态保存与重载 | `state_dict`、`jt.save/load` 和导出参数让训练、推理、哈希校验使用同一参数语义 | 检查点与复现收据 |

NumPy、Pandas 和 Numba 负责 CSV 解析、图统计、索引与稳定序列化；可学习参数、
自动求导、优化器、损失函数和 GPU 前向由 Jittor 执行。

下面是 final MF 使用的 Jittor 前向核心，OOV 节点会被显式 mask：

```python
def execute(self, source, candidates):
    source_known = (source > 0).unsqueeze(1).unsqueeze(2)
    candidate_known = (candidates > 0).unsqueeze(2)
    source_vector = self.source(source).unsqueeze(1) * source_known
    item_vector = self.item(candidates) * candidate_known
    score = (source_vector * item_vector).sum(dim=2)
    return (
        score
        + self.item_bias(candidates).squeeze(2) * (candidates > 0)
    )
```

集合模型则直接使用 Jittor 多头注意力、残差连接和 LayerNorm：

```math
\begin{aligned}
H'  &= \mathrm{LN}(H+\mathrm{MHA}(H,H,H)),\\
H'' &= \mathrm{LN}(H'+\mathrm{FFN}(H')).
\end{aligned}
```

源码证据：

- A 榜图排序：[`dataset1/run.py`](A/code/raw_training/dataset1/run.py)
- A 榜 VAE/BPR：[`train_vae_jittor.py`](A/code/raw_training/dataset2/train_vae_jittor.py) /
  [`train_bpr_jittor.py`](A/code/raw_training/dataset2/train_bpr_jittor.py)
- B 榜时序注意力：[`temporal_attention_jittor.py`](B/code/pipeline/code/b_rank/temporal_attention_jittor.py)
- B 榜 Set Transformer：[`d3_set_transformer_v49.py`](B/code/pipeline/code/ruc3/d3_set_transformer_v49.py)
- B 榜元排序器：[`train_hierarchy_jittor.py`](B/code/pipeline/code/third_1/train_hierarchy_jittor.py)
- B 榜 MF 训练：[`implicit_mf_jittor.py`](B/code/pipeline/code/b_rank/implicit_mf_jittor.py)

<a id="a-list"></a>

## 3. A 榜具体方案

A 榜包含 Dataset1 和 Dataset2。记录结果为 **第 7 名，
`1.521072794155721`**。详细运行协议见 [`A/README.md`](A/README.md)。

### 3.1 Dataset1：时序图特征与双成员排序

Dataset1 先对训练边做稳定时间排序，只使用查询以前的历史构造四组特征：

| 特征组   | 内容                                          |
| -------- | --------------------------------------------- |
| 节点统计 | 来源/目标累计次数、入度、出度、首次与最近交互 |
| 二元关系 | 来源—目标频次、最近一次交互、候选池出现频率  |
| 局部图   | 最近邻居、入/出邻居重叠、共同邻居             |
| 时间     | 小时、星期、时间间隔与单调新近度              |

基础 `Net` 使用 `dim -> 128 -> 64 -> 1` 的 Jittor MLP，同时学习来源/目标
embedding 和偏置：

```math
s_{\mathrm{D1}}(s,c)=
\mathrm{MLP}(x_{s,c})
+\langle e_s,e_c\rangle+b_s+b_c.
```

生产双成员实际调用 `Net(use_hist=False, use_cf=True)`：近期性进入手工时序
特征，额外五个二部图邻居重叠特征补充协同关系；文件中定义的 `NetAttn`
不属于该生产调用链。训练样本固定为“1 个正目标 + 99 个候选池负例”，在每个
候选组内计算交叉熵。两个固定种子 `20260705`、`20260715` 独立训练后集成。

记录结果的 Dataset1 后处理使用同一来源的跨查询支持。只有最大支持至少为 4、
且领先第二名至少 2 行时才触发；其余候选的相对顺序不变。

### 3.2 Dataset2：稀疏偏好与候选集合建模

Dataset2 将时间衰减后的历史表示为 CSR 稀疏矩阵，再从互补方向训练成员：

| 成员                   | 作用                               | 核心目标                           |
| ---------------------- | ---------------------------------- | ---------------------------------- |
| MultVAE / RecVAE       | 重构来源的全局稀疏偏好             | 重构损失 + KL 约束                 |
| BM25-BPR               | 学习正目标高于负目标               | `softplus(-(positive-negative))` |
| pool / set             | 建模候选池统计和置换不变集合上下文 | 100 候选组内分类                   |
| multislice Transformer | 融合多个历史时间切片               | 候选集合注意力                     |
| warm residual          | 补充热节点行的局部误差             | 低幅度候选残差                     |

锁定结果在基座上加入两种确定性结构信号：

1. 同一 `(src, time)` 精确组中的跨行候选支持；
2. 同时刻、同候选的跨来源 BPR32 社区相似度。

```math
z=
\mathrm{qnorm}(\log p_{\mathrm{base}})
+0.05\,\mathrm{qnorm}(I_{\mathrm{exact}})
+0.02\,\mathrm{qnorm}(I_{\mathrm{community}}).
```

```math
p=\mathrm{softmax}_{100}(z).
```

### 3.3 A 榜输出

Dataset1/2 最终写入候选概率；每行保持 100 列、数值有限、候选位置不变。
构建器检查官方数据、基座、BPR32、规则文件、CSV 成员、ZIP CRC 和最终哈希。

<a id="b-list"></a>

## 4. B 榜具体方案

B 榜包含 Dataset3 和 Dataset4。记录结果为 **第 2 名，
`1.5240999401892983`**。它保留 A 榜的角色分层：历史编码负责读取过去，
个体偏好成员负责候选打分，集合成员负责候选关系，最后在候选列内校准输出。
具体网络按 B 榜字段、规模和历史跨度实例化，并非复用同一组参数或同一个模型
文件。详细技术说明见 [`B/README.md`](B/README.md)。

### 4.1 Dataset3：九成员图集成与结构链

Dataset3 是 A 榜 Dataset1 图排序方向的数据适配。基座训练
`raw / cf / hist_cf x 3 seeds` 共九个 Jittor 成员：

| 变体        | 数据信号                                          |
| ----------- | ------------------------------------------------- |
| `raw`     | 通用时序图统计、来源—目标 embedding              |
| `cf`      | 增加 5 个二部图邻居重叠特征                       |
| `hist_cf` | 在 CF 特征上增加近期目标 embedding 的 masked mean |

九个训练目录进一步产生 base、FastRanker 和传播 embedding 分数，并与四个
独立启发式组成 31 路候选。`meta_train` 拟合凸组合，`validation` 选择组合
或单分量，`confirmation` 独立确认。

| 阶段 | 数据适配内容                        | 固定策略                                  |
| ---- | ----------------------------------- | ----------------------------------------- |
| C2   | 同时刻跨来源、同来源`+-300s` 支持 | 权重`0.10`、`0.05`                    |
| C3   | 未见 pair 的多尺度方向支持          | `[-0.10, 0.225, 0.30, 0.305]`           |
| C5   | `(900s, 86400s]` session ring     | `[0.2625, 0.28, -0.0525]`               |
| C6   | 并列未见最大值                      | tie scale`0.20`                         |
| RUC4 | 100 候选集合上下文                  | 3 seeds、2 blocks、4 heads、scale`0.30` |

最后只加入来自官方 Dataset3 历史的低幅度目标频次项：

```math
r_{\mathrm{pop}}=
\tanh\left(
\frac{\mathrm{qnorm}(\log(1+\mathrm{count}(c)))}{2}
\right).
```

```math
\mathrm{score}_3=\mathrm{base}_3+0.005r_{\mathrm{pop}}.
```

### 4.2 Dataset4：时序/MF 专家与 75 特征元排序

Dataset4 是 A 榜 Dataset2 稀疏交互方向的数据适配。它保留多专家与候选集合
融合框架，按更长历史和更大数据规模配置成员：

| 专家族               |                     数量 | 数据作用                       |
| -------------------- | -----------------------: | ------------------------------ |
| history temporal     | `h32 x 3`、`h64 x 3` | 候选条件的近期历史注意力       |
| test-pool temporal   |                    `3` | test-pool 回放下的时序偏好     |
| implicit MF          |                    `3` | 来源—目标长期兼容性           |
| transition-MF        |                    `1` | 来源转移模式                   |
| pair-new Transformer |                    `6` | hidden`64/96` 的候选集合残差 |

候选 `c` 对历史节点 `h_j` 的注意力同时考虑 embedding 相似度和真实时间间隔：

```math
a_{c,j}=
\mathrm{softmax}_j
\left(
\frac{\langle q(c),k(h_j)\rangle}{\sqrt d}
-\tau\Delta t_j
\right).
```

RUC4 再构造 `history` / `test_pool` replay cache，训练三种子 session-graph
hard ranker。`third_1` 将 replay、identity、baseline、temporal、MF、
transition-MF、hierarchy 和 neighbor 合并为 75 个特征。

元排序器分开编码前 22 个 full-history 特征与其余 53 个 recent/relational
特征，再加入整行候选均值：

```python
full = self.full(values[:, :, :22])
recent = self.recent(values[:, :, 22:])
local = self.local(jt.concat((full, recent), dim=2))
context = local.mean(dim=1, keepdims=True)
context_shape = (
    local.shape[0],
    local.shape[1],
    local.shape[2],
)
context = context.broadcast(context_shape)
joined = jt.concat((full, recent, context), dim=2)
score = self.output(joined).squeeze(-1)
```

训练目标联合 listwise、hard-negative 和 soft-rank：

```math
\mathcal{L}=
0.50\,\mathcal{L}_{\mathrm{list}}
+0.30\,\mathcal{L}_{\mathrm{hard}}
+0.20\,\mathcal{L}_{\mathrm{soft\_rank}}.
```

推理融合 full、reverse 和 no-recent 三个视图：

```python
meta_residual = qnorm(
    0.40 * full
    + 0.40 * reverse
    + 0.20 * no_recent
)
```

最后独立训练 32 维 implicit MF，并以 `0.02` 有界残差加入 D4 基座：

```math
f_{\mathrm{MF32}}(s,c)=\langle u_s,v_c\rangle+b_c,
```

```math
\mathrm{score}_4=
\mathrm{base}_4+
0.02\tanh
\left(
\frac{1}{2}\mathrm{qnorm}(f_{\mathrm{MF32}})
\right).
```

稳定降序后映射到 `linspace(1, 0, 100)`，再写回原候选列。

### 4.3 A/B 榜如何保持基本一致

| 共同角色 | A 榜实现 | B 榜数据方向适配 |
| --- | --- | --- |
| 因果历史表示 | 时间图统计、CSR 稀疏历史 | 更大实体索引、更长窗口和流式 cache |
| 个体偏好打分 | D1 图排序、D2 VAE/BPR | D3 Net/FastRanker、D4 Temporal/MF |
| 候选关系建模 | D2 pool/set/Transformer | D3 Set Transformer、D4 Pair/session/meta |
| 候选内校准 | D1 来源支持、D2 `qnorm` 残差 | D3 结构链、D4 gated residual 与稳定重排 |
| 输出契约 | Dataset1/2 概率 | Dataset3 概率、Dataset4 rank grid |

“基本一致”指任务契约、角色分层、Jittor 实现和候选列边界一致；具体成员依据
字段、图密度、历史跨度和数据规模选择。B 榜新增的 Set/session/meta 成员属于
这些角色的数据化实例，而不是修改预测目标或引入新的标签。

<a id="reproduce"></a>

## 5. 复现入口

先理解赛题和算法，再选择对应榜单的执行路径。

### A 榜

```bash
# 快速重建记录结果
python A/code/main.py verify \
  --data /path/to/data_A.zip \
  --output /path/to/a-verify

# 从官方数据训练 raw Jittor 成员并执行 fresh 推理
python A/code/main.py raw \
  --data /path/to/data_A.zip \
  --models /path/to/a-models \
  --output /path/to/a-fresh
```

### B 榜

```bash
# 快速重建记录结果
python B/code/main.py verify \
  --data /path/to/data_B.zip \
  --output /path/to/b-verify

# 官方数据 -> 全部训练 -> fresh 推理 -> 竞赛级重排 -> 提交
python B/code/main.py reproduce \
  --data /path/to/data_B.zip \
  --output /path/to/b-reproduce
```

默认继承调用者的 `CUDA_VISIBLE_DEVICES`；若未设置，则使用首个可见 GPU。
只有需要主动选择设备时才追加 `--gpu N`，该编号属于运行者自己的机器。

| 路径           | 训练                     | 主要用途                           |
| -------------- | ------------------------ | ---------------------------------- |
| A`verify`    | 不重训                   | 快速复验 A 榜记录结果              |
| A`raw`       | 重训 A 榜 raw 成员       | 审阅训练与 fresh 推理              |
| B`verify`    | 不重训                   | 快速复验 B 榜记录结果              |
| B`reproduce` | 重训 D3/D4 与 final MF32 | 打通官方数据到最终提交的完整调用链 |

`B reproduce` 满足完整提交口径：代码独立从 `data_B.zip` 的原始训练数据
完成 D3、D4 与 final MF32 训练，再使用测试候选生成 fresh 预测。固定的
竞赛级分数空间重排用于消除机器和算子差异，少量缺失参数按冻结竞赛状态
对齐，随后从 fresh 结果直接生成新的 `frozen_base.ckpt` 和最终提交。
fresh 产物是强制输入，不会被快速复现链的冻结权重覆盖。详细边界见
[`B/README.md#reproducibility-contract`](B/README.md#reproducibility-contract)。

最终 B 榜 `result.zip` SHA-256：

```text
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

## Repository

```text
.
├── A/                  # A 榜复现、raw training 与技术说明
├── B/                  # B 榜双链路、完整训练图与技术说明
├── LICENSE
├── NOTICE.md
└── README.md
```

当前执行入口和算法口径以本 README、`A/B/code/main.py` 与实际源码为准；
[`A/提交说明文档.pdf`](A/提交说明文档.pdf) 和
[`B/提交说明文档.pdf`](B/提交说明文档.pdf) 是比赛提交时的历史材料。

<details>
<summary>GIF 来源与授权</summary>

- [Collaborative filtering](https://commons.wikimedia.org/wiki/File:Collaborative_filtering.gif)，Moshanin。
- [Barabási–Albert model](https://commons.wikimedia.org/wiki/File:Barabasi_Albert_model.gif)，Harp。
- [Social graph](https://commons.wikimedia.org/wiki/File:Social_graph.gif)，Festys。

以上资源来自 Wikimedia Commons，采用 CC BY-SA 3.0；仅用于说明图关系，
不是模型训练或推理资产。

</details>

## License

Repository code is released under the MIT License. Competition datasets and
third-party dependencies follow their original licenses and distribution rules.
