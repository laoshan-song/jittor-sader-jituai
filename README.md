# 🧭 jittor-sader-jituai

> 一条边发生之后，下一条边会走向哪里？
>
> 这是 sader 团队留下的时序图推荐实验记录：赛道一 A 榜第 7 名，B 榜第 2 名。代码、模型、报告和踩坑心得都放在这里。

[![Jittor](https://img.shields.io/badge/Framework-Jittor-0ea5e9?style=flat-square)](https://github.com/Jittor/jittor) [![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=flat-square)](https://www.python.org/) [![Track](https://img.shields.io/badge/Task-Temporal%20Graph%20Recommendation-8b5cf6?style=flat-square)](#-ab-榜算法说明与一致性)

<p align="center">
  <a href="https://commons.wikimedia.org/wiki/File:Collaborative_filtering.gif">
    <img src="https://upload.wikimedia.org/wikipedia/commons/5/52/Collaborative_filtering.gif" alt="协同过滤动图" width="760">
  </a>
</p>

一句话概括：给定来源、时间和 100 个候选，我们让历史交互先说话，再让候选集合内部互相比较，最后输出一份可以复验的结果。

| 目录 | 这里放什么 | 适合先读什么 |
| --- | --- | --- |
| `A/` | A 榜完整复现包、训练源码、冻结工件和技术报告 | `A/README.md` |
| `B/` | B 榜复现包、完整训练链路、流式推理和技术报告 | `B/README.md` |

B 榜两个公开入口：

```bash
python B/code/main.py verify --data /path/to/data_B.zip --output /path/to/verify --gpu 0
python B/code/main.py reproduce --data /path/to/data_B.zip --output /path/to/reproduce --gpu 0
```

官方数据和最终结果包不随仓库提供；复现所需的源码、模型状态与数值一致性资产按各榜 `MANIFEST.sha256` 纳入审计。

## 📝 参赛者手记

我们一开始只想把分数做高。很快发现，图推荐的难点藏在“时间”两个字里。

一条未来发生的边，悄悄混进历史统计，验证分数会漂亮，线上表现会失真。于是我们把时间边界写进代码，把历史、候选和输出一层层锁住。

分数相加很快。尺度对齐很难。VAE、BPR、图模型各自有自己的脾气，候选内 `qnorm` 让它们先站到同一把尺子上，再谈融合。

热门候选很有诱惑力。我们给它的权限很小：低幅度残差，只在当前 100 个候选里生效。模型负责学习，规则负责校正，审计负责刹车。

最后一个教训来自 ZIP。内容相同，压缩时间戳不同，SHA-256 也会不同。比赛交付的最后一公里，同样需要算法思维：固定顺序、固定小数位、固定 CRC，结果才真正可复现。

这份仓库更像一张比赛地图。A 榜提供 `verify` 与 raw training；B 榜并列提供快速 `verify` 和完整 `reproduce`，各入口都先执行对应的包审计。

## 🫧 图推荐动图角落

读到这里，脑子里大概已经有一张图了。再看两张会动的。

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://commons.wikimedia.org/wiki/File:Barabasi_Albert_model.gif">
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/48/Barabasi_Albert_model.gif" alt="Barabasi Albert 网络生长" width="440">
      </a>
      <br>
      <sub><b>Barabási–Albert model</b>：新节点偏爱连接“已经很热”的节点。</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://commons.wikimedia.org/wiki/File:Social_graph.gif">
        <img src="https://upload.wikimedia.org/wikipedia/commons/d/de/Social_graph.gif" alt="社交图逐步展开" width="440">
      </a>
      <br>
      <sub><b>Social graph</b>：人、内容、行为，慢慢连成一张网。</sub>
    </td>
  </tr>
</table>

动图来自 [Wikimedia Commons](https://commons.wikimedia.org/)，点击图片可查看原始文件与授权信息：
[Moshanin 的 Collaborative filtering](https://commons.wikimedia.org/wiki/File:Collaborative_filtering.gif)、
[Harp 的 Barabási–Albert model](https://commons.wikimedia.org/wiki/File:Barabasi_Albert_model.gif)、
[Festys 的 Social graph](https://commons.wikimedia.org/wiki/File:Social_graph.gif)，均为 CC BY-SA 3.0。它们和仓库里的模型没有直接关系，只负责把“邻居变多、关系变密、候选互相影响”这件事演给你看。

## 🧩 A/B 榜算法说明与一致性

下面的说明根据 [A 榜技术报告](A/提交说明文档.pdf) 和 [B 榜技术报告](B/提交说明文档.pdf) 整理，代码位置以两个比赛目录中的实际实现为准。

### 1. 共同的问题定义

赛道一是一个时序交互图上的候选排序任务。对每条查询，官方数据给出来源节点 `s`、查询时刻 `t`（由需要时间信息的数据集使用）和固定的 100 个候选节点 `c1...c100`。模型不负责从全体节点中召回候选，只负责回答：在已经给定的 100 个候选中，哪个更可能是目标，并输出与候选列一一对应的分数或概率。

两榜都遵循同一条数据边界：

- 训练和统计只使用官方历史交互，并按时间先后构造历史，避免未来边进入当前查询的特征。
- A 榜的快速路径读取保留状态，raw training 单独提供；B 榜的全链路路径重新训练模型并保留 fresh 中间产物。
- 不读取测试集真实标签，不使用外部数据，也不把候选集合之外的节点引入单条查询的比较。

模型输出通常先是未归一化分数。为了让不同模型的数值尺度可以融合，代码在每一行的 100 个候选内部做标准化：

```text
qnorm(x) = (x - mean(x)) / (std(x) + 1e-6)
```

这里的均值和标准差只由当前查询的候选计算，不会在不同查询之间传播信息。融合后再根据数据集接口处理：A 榜的概率成员使用候选内 `softmax`，保证每行概率和为 1；B 榜 Dataset4 则使用固定的 rank grid 表达名次。两种方式都保持候选列对齐、输出宽度固定和数值有限。

两榜都提供历史提交的 SHA-256 验证。B 榜对外只有两个命令：`verify` 用保留的推理状态快速复现，`reproduce` 从 `data_B.zip` 完整执行训练、fresh 推理、数值一致性处理和提交构建。下面第 3 节按 Dataset3/Dataset4 展开。

### 2. A 榜：多成员时序图排序 + 候选内结构校正

#### 2.1 Dataset1：时序图特征和双模型集成

Dataset1（实现见 `A/code/raw_training/dataset1/run.py`）先将训练边按时间稳定排序，并只用历史段建立图统计。对每个来源、目标和来源—目标对，维护累计次数、最近出现时间、首次/最近交互、最近邻居序列，以及小时和星期等周期特征；对当前查询还计算候选池频次、局部邻居重叠和来源—候选对频次。时间新近度使用单调衰减形式 `1 / (1 + log1p(delta / 86400))`，让近期交互影响更强，但不会产生无界数值。

基础 `Net` 模型把这些统计特征与来源/目标 embedding、来源/目标偏置拼接，送入 Jittor 的多层 MLP（特征维度 → 128 → 64 → 1）。它同时学习两类信息：图统计描述“当前来源在当前时刻的局部状态”，embedding 内积描述“这个来源和候选之间较稳定的兼容性”。

`NetAttn` 在同一主干上增加短期兴趣分支：当前候选 embedding 作为 query，来源最近交互过的目标 embedding 作为 key/value，并把真实时间间隔作为衰减项；无效历史位置会被 mask，空历史不会被伪造成某个偏好。这样不同候选可以关注来源近期历史中的不同节点，而不是简单平均所有邻居。

训练样本固定为“1 个正目标 + 99 个候选池负例”的 100 候选组，在组内计算交叉熵，目标是让真实目标相对同组候选排在前面，而不是把所有查询混成一个全局分类问题。协调器用两个固定种子（`20260705`、`20260715`）训练两个成员，再在打分阶段集成，降低单次随机初始化带来的排序波动。

#### 2.2 Dataset2：稀疏历史、多种排序成员和基座

Dataset2（训练实现位于 `A/code/raw_training/dataset2/`）的历史交互先按时间端点切片，并使用指数时间衰减。稀疏表示采用 CSR 风格的 `indptr/indices/values` 数组，训练 batch 才展开成 Jittor 张量。

- **MultVAE/RecVAE**：把每个来源的历史交互看成稀疏偏好向量，学习隐变量重构和目标偏好。MultVAE 使用重参数化采样和 KL 退火；RecVAE 使用更深的残差编码器和带历史先验的 KL 约束。
- **BM25-BPR**：先对历史交互做时间衰减和 BM25 加权，再用来源/目标 embedding 做成对排序。每个正目标配 8 个负目标，优化 `softplus(-(positive - negative))`，直接学习“正目标应高于负目标”的相对关系。
- **pool、set、多切片 set、Transformer**：输入仍是每行 100 个官方候选，但分别建模候选池特征、无序集合上下文和多个历史时间切片。它们学习的是“同一候选集合内哪些组合更有辨别力”，不会扩展候选域。
- **warm residual**：只在适用的热节点行上学习一个候选内残差信号，作为基座的补充，而不是把热门程度直接当成答案。

这些成员各自保存模型和元数据，随后按固定顺序合成为 Dataset1/Dataset2 基座。记录 A 榜结果时不重新训练全部成员，而是读取冻结的基座概率和一个 32 维 Jittor BPR 检查点，进入确定性的后处理路径。

#### 2.3 A 榜的锁定后处理

Dataset1 的锁定后处理（`A/code/dataset1/d1_source_support_postprocess.py`）使用**同一来源的跨行支持**：统计某个候选在该来源的其他测试查询中出现了多少次，选出支持最多的候选。只有最大支持数至少为 4、且领先第二名至少 2 行时才触发规则；如果这个候选还不是基座的 top-1，就在 logit 空间把它提升到当前最大 logit 以上 1.0，再对该行重新 softmax。该操作只改变明确满足条件的胜者，不改变其余 99 个候选的相对次序。

Dataset2 的第一个残差（`A/code/dataset2/d2_exact_group_postprocess.py`）使用精确 `(src, time)` 查询组：对同一来源、同一时刻的其他行，判断某候选是否出现过，并按“其他行是否支持”得到二值信号。它不是按重复次数无限加分，而是先在候选内 `qnorm`，再以固定权重 `0.05` 加到 `qnorm(log(base_probability))` 上。

第二个残差是**跨来源社区信号**。代码只在同一时间、同一候选的跨来源用户对上计算 32 维 BPR 用户向量的 Jittor 余弦相似度，排除同来源和未知用户，只把余弦最高的 10% 标记为社区关系。该二值标记再次在每行候选内 `qnorm`，以固定权重 `0.02` 加入分数。Dataset2 的锁定公式可以概括为：

```text
z = qnorm(log(base_probability))
    + 0.05 * qnorm(exact_other_row_support > 0)
    + 0.02 * qnorm(cross_source_community_top10pct)
p = softmax(z)              # 仅在当前行的 100 个候选内
```

最终固定输出 `dataset1.csv` 和 `dataset2.csv`；锁定构建器要求它们分别为 61,051 × 100 和 153,420 × 100，并校验概率有限性、ZIP CRC、成员哈希及整个 `result.zip` 的 SHA-256。A 榜报告中的“精确复现”指这条冻结工件 + 固定后处理路径，并不表示从零训练能够得到完全相同的随机模型参数。

### 3. B 榜：Dataset3/Dataset4 完整训练

#### 3.1 双链路

| 路径 | 计算内容 | 用途 |
| --- | --- | --- |
| `verify` | 保留推理状态 + 固定后处理 | 快速字节级复现 |
| `reproduce` | D3/D4 全部训练阶段 + fresh 推理 + 数值一致性 + 提交构建 | 审阅和执行完整训练链 |

`reproduce` 由 `B/code/pipeline/reproduce_full.py` 驱动，调用顺序为
`C2 → C3 → C5 → C6 → RUC4 → third_1 → final MF32`。所有训练特征来自官方历史，不读取测试标签。

#### 3.2 Dataset3

Dataset3 首先训练 `raw/cf/hist_cf × 3 seeds` 的九成员基础集成，每个成员同时训练 scene model 与 FastRanker。随后依次加入 C2 source/session 支持、C3 多尺度方向支持、C5 session-ring、C6 固定 tie-group 变换和三种子 Jittor Set Transformer。`third_1` 保留 RUC4 的 D3 输出，最终构建器只增加低幅度目标频次残差：

```text
score3 = base3 + 0.005 * tanh(qnorm(log(1 + destination_count)) / 2)
```

#### 3.3 Dataset4

Dataset4 的 C2 层训练 32/64 历史长度 temporal、test-pool temporal、三成员隐式 MF、transition-MF 和六成员 pair-new Transformer。RUC4 再训练 session-graph hard ranker并完成 RP3/RUC2/RUC3/RUC4 候选融合。`third_1` 构建 replay、identity、hierarchy、neighbor 与 baseline 特征，训练 75 特征 Jittor 元排序器；最后独立训练 32 维 MF：

```text
f(s, c) = u_s · v_c + b_c
score4 = base4 + 0.02 * tanh(qnorm(f(s, c)) / 2)
```

#### 3.4 数值一致性

历史算子版本、浮点环境和少量未保留的中间参数状态会造成确定性差异。全链路在 fresh D3/D4 分数和 fresh MF32 已生成之后，才在定点/q7/q8 表示上执行固定数值一致性处理；该层不替代训练阶段，也不读取快速路径的 `assets/locked`。若 fresh 来源哈希偏离已记录运行，程序直接失败，不会静默接受。

最终稳定排序并写出 `dataset3.csv` 与 `dataset4.csv`，`result.zip` 必须命中：

```text
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

### 4. 两榜如何保持同一架构

| 层次 | A 榜在对应数据集上的实现 | B 榜在对应数据集上的实现 | 共同架构 |
| --- | --- | --- | --- |
| 数据边界 | 历史交互、来源、时间和 100 候选 | 历史交互、来源和 100 候选，按数据集需要使用时间 | 不读测试标签、不引入外部数据 |
| 学习/统计成员 | 按 Dataset1/2 的字段实例化图排序、VAE/BPR 和集合成员 | C2/C3/C5/C6/RUC4/third_1 多专家与元融合链路 | 都从官方历史中学习或统计 |
| 融合方式 | 候选内 `qnorm`、固定残差、softmax | 候选内 `qnorm`、有界残差、固定 rank grid | 只在当前 100 个候选内校准 |
| 工程落地 | 冻结基座和 BPR32 的精确重建 | 全链路重跑 + fresh 数值/参数域适配 | 固定接口、稳定序列化和哈希审计 |

所以，“算法一致”指两榜共享任务边界、候选内排序思想、Jittor 实现和可审计输出契约。由于官方数据集的字段、实体规模和查询接口不同，具体成员与批处理方式做相应适配；这属于同一架构下的数据集落地，不是架构改写。

## 🗂️ 仓库结构

```text
.
├── A/                            # A 榜比赛包
├── B/                            # B 榜比赛包
├── LICENSE
├── NOTICE.md
└── README.md
```

## 📜 License

Repository code is released under the MIT License. Competition datasets and
third-party dependencies follow their original licenses and distribution rules.
