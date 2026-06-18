# jittor-sader-jituai

第六届计图人工智能挑战赛项目仓库。当前包含热身赛一和正式赛道一：

- 热身赛一：基于 Cora 引文网络的 GCN 节点分类
- 正式赛道一：基于图学习的动态推荐任务

## Environment

- Python: 3.10
- Framework: Jittor 1.3.11.0
- Graph library: JittorGeometric 2.0.0

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/AlgRUC/JittorGeometric.git
python scripts/patch_jittor_geometric_cpu.py
source env.sh
python verify_env.py
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the CPU compatibility notes.

## Data Preparation

Warm-up 1: download the release package from the competition platform and place
the dataset at:

```text
warmups/cora_gcn/data/cora.pkl
```

The dataset file is not tracked in Git. The expected fields are described in
[warmups/cora_gcn/README.md](warmups/cora_gcn/README.md).

Track 1: place the official A leaderboard data zip outside this repository:

```text
../data_A.zip
```

The zip file should contain scene folders such as `dataset1/` and `dataset2/`,
each with `train.csv` and `test.csv`. See
[formal/track1_dynamic_recommendation/README.md](formal/track1_dynamic_recommendation/README.md).

## Training

Run the warm-up training script:

```bash
source env.sh
python warmups/cora_gcn/gcn.py \
  --data-path warmups/cora_gcn/data/cora.pkl \
  --output warmups/cora_gcn/result.json \
  --seed 42 \
  --epochs 200
```

The script trains a two-layer GCN and reports training accuracy and best
validation accuracy.

Run the Track 1 heuristic baseline:

```bash
python formal/track1_dynamic_recommendation/baseline.py \
  --data-zip ../data_A.zip \
  --output outputs/track1/result.zip
```

It scores each row's 100 candidate target nodes with historical interaction,
recency, and popularity features.

## Evaluation And Inference

The competition warm-up release evaluates the generated `result.json` on the
hidden test labels. To regenerate the prediction file only after training, run
the same command above and package the result:

```bash
cd warmups/cora_gcn
python gcn.py --seed 42 --epochs 200 --output result.json
zip ../../submissions/warmup1-result.zip result.json
```

`result.json` and submission archives are generated artifacts and are ignored by
Git.

For Track 1, submit the generated `outputs/track1/result.zip`. It contains one
CSV file per scene, for example `dataset1.csv` and `dataset2.csv`. Each row has
100 probabilities in the same order as the corresponding test candidates.

## Results

- Task: warm-up 1, Cora node classification
- Metric: accuracy on node labels
- Local best validation accuracy: 0.8120
- Platform submission status: passed

Track 1 uses MRR on candidate rankings. The baseline is intended to produce a
valid first submission and a reproducible starting point for stronger graph
models.

The local validation score may differ slightly across machines because Jittor,
CPU/GPU kernels, and random initialization can vary. Use `--seed` to keep runs
as reproducible as possible.

## Repository Layout

```text
.
├── docs/                         # competition notes
├── formal/
│   └── track1_dynamic_recommendation/
│       ├── baseline.py           # Track 1 submission baseline
│       └── README.md             # Track 1 task notes
├── scripts/                      # environment/helper scripts
├── submissions/                  # generated submission archives, ignored
├── warmups/
│   └── cora_gcn/
│       ├── data/README.md        # data placement instructions
│       ├── gcn.py                # training and inference entry point
│       └── README.md             # task-specific notes
├── ENVIRONMENT.md
├── LICENSE
├── NOTICE.md
├── env.sh
├── requirements.txt
└── verify_env.py
```

## License

Repository code is released under the MIT License. Competition datasets and
third-party dependencies follow their original licenses and distribution rules.
