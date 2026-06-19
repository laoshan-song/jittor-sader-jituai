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
python formal/track1_dynamic_recommendation/train_jittor_mf.py \
  --data-zip ../data_A.zip \
  --output outputs/track1/result.zip \
  --epochs 6 \
  --mf-weight 1 \
  --mf-gate repeat-pair \
  --probability-mode rank
```

输出文件结构：

```text
result.zip
├── dataset1.csv
└── dataset2.csv
```

生成的 `outputs/`、`*.csv` 和 `*.zip` 文件默认不会提交到 Git。

`train_jittor_mf.py` 是正式复现入口，基于 Jittor 训练 BPR
matrix-factorization reranker，并融合历史/序列统计特征生成提交文件。
`baseline.py` 是无需训练的规则 baseline，`train_mf_rerank.py` 仅作为
早期 PyTorch 探索脚本保留，不作为代码审核复现主路径。

快速自检命令：

```bash
python formal/track1_dynamic_recommendation/train_jittor_mf.py \
  --data-zip ../data_A.zip \
  --output outputs/track1/jittor_smoke.zip \
  --scenes dataset1 \
  --epochs 1 \
  --dim 8 \
  --batch-size 1024 \
  --limit-train-rows 10000 \
  --limit-test-rows 100 \
  --cpu
```

服务器无外网时，Jittor 首次 CUDA 编译可能需要离线补齐 cutt 缓存。推荐将
`JITTOR_HOME`、`XDG_CACHE_HOME`、模型和输出都放在数据盘目录，避免写入 home。
当前复现服务器使用如下环境变量绕过无外网 cutt 和缺失 cuDNN 头文件的问题：

```bash
CUDA_VISIBLE_DEVICES=0 \
HOME=/data1/ml-1-1 \
XDG_CACHE_HOME=/data1/ml-1-1/cache \
JITTOR_HOME=/data1/ml-1-1/jittor_home \
use_cutt=0 \
conv_opt=1 \
/data1/ml-1-1/venv/bin/python formal/track1_dynamic_recommendation/train_jittor_mf.py \
  --data-zip data_A.zip \
  --output outputs/result_jittor_mf_repeat_pair.zip \
  --scenes dataset1,dataset2 \
  --epochs 6 \
  --dim 96 \
  --batch-size 65536 \
  --lr 0.03 \
  --seed 2026 \
  --mf-weight 1 \
  --mf-gate repeat-pair \
  --probability-mode rank
```

## Local Evaluation

本地评估分为两层，避免只看一个会过拟合的 proxy：

1. `evaluate_submission.py` 审计已生成的提交包。它不需要标签，不估计线上
   MRR，而是检查格式、概率塌缩、排序熵、与稳定 baseline 的 top1/top10 差异。
2. `offline_eval.py` 从 `train.csv` 内部构造时间切分和候选集，计算带标签的
   MRR/Hit@K。这个指标只作为代理分数，需要同时看不同 split 和 candidate
   strategy。

提交包审计示例：

```bash
python formal/track1_dynamic_recommendation/evaluate_submission.py \
  --data-zip ../data_A.zip \
  --baseline-zip outputs/track1/result_sequence.zip \
  --submissions \
    outputs/track1/result_sequence.zip \
    outputs/track1/result_mf_conservative_rank_w1.zip \
    outputs/track1/result_mf_conservative_rank_w2.zip \
    outputs/track1/result_mf_final.zip \
    outputs/track1/result_mf_hardneg_final.zip \
  --output-json outputs/track1/submission_audit.json
```

离线代理 MRR 示例：

```bash
python formal/track1_dynamic_recommendation/offline_eval.py \
  --data-zip ../data_A.zip \
  --sample-positives 1000 \
  --splits temporal,official \
  --candidate-strategies mixed,hard \
  --output-json outputs/track1/offline_eval_core_sample.json
```

注意：`dataset2` 的 official split 与正式测试候选分布不同，曾经把 hard-negative
MF 的本地验证推高但线上变差。因此本地 MRR 不能单独作为提交依据。
