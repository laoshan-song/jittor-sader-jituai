# Track 1: Dynamic Recommendation

赛道一是基于图学习的动态推荐任务。训练集每行是一条带时间戳的交互边：

```text
src,dst,time
```

含义为源节点 `src` 和目标节点 `dst` 在时间 `time` 发生交互。测试集每行给出
一个源节点、一个时间戳和 100 个候选目标节点：

```text
src,time,c1,c2,...,c100
```

提交时需要为每一行的 100 个候选目标节点分别输出一个交互概率。概率顺序必须
和测试集候选节点顺序一致，每行 100 个数，保留 8 位小数。

## Baseline

`baseline.py` 是一个无需训练神经网络的可提交 baseline。它根据历史动态图统计：

- 源节点和候选目标节点的历史交互次数
- 源节点和候选目标节点最近一次交互距离当前时间的远近
- 候选目标节点的全局热度
- 候选目标节点最近一次被交互的时间

这些分数会被转换为 100 个候选节点上的概率分布，用于生成 A 榜提交包。

## Data

将官方 A 榜数据包放在仓库外或仓库内均可，推荐保持在仓库外：

```text
/home/laoshansong/M L/data_A.zip
```

压缩包结构应类似：

```text
data_A.zip
├── dataset1/train.csv
├── dataset1/test.csv
├── dataset2/train.csv
└── dataset2/test.csv
```

## Generate Submission

从仓库根目录运行：

```bash
python formal/track1_dynamic_recommendation/baseline.py \
  --data-zip ../data_A.zip \
  --output outputs/track1/result.zip
```

输出文件结构：

```text
result.zip
├── dataset1.csv
└── dataset2.csv
```

生成的 `outputs/`、`*.csv` 和 `*.zip` 文件默认不会提交到 Git。
