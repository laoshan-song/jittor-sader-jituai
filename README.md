# jittor-sader-jituai

第六届计图人工智能挑战赛项目仓库。当前完成内容为热身赛一：
基于 Cora 引文网络的 GCN 节点分类。

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

Download the warm-up 1 release package from the competition platform and place
the dataset at:

```text
warmups/cora_gcn/data/cora.pkl
```

The dataset file is not tracked in Git. The expected fields are described in
[warmups/cora_gcn/README.md](warmups/cora_gcn/README.md).

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

## Results

- Task: warm-up 1, Cora node classification
- Metric: accuracy on node labels
- Local best validation accuracy: 0.8120
- Platform submission status: passed

The local validation score may differ slightly across machines because Jittor,
CPU/GPU kernels, and random initialization can vary. Use `--seed` to keep runs
as reproducible as possible.

## Repository Layout

```text
.
├── docs/                         # competition notes
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
