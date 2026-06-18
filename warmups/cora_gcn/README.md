# Warm-up 1: Cora GCN

This warm-up trains a two-layer GCN on the Cora citation graph with Jittor and
JittorGeometric, then writes test-node predictions to `result.json`.

## Data

Place the competition release dataset here:

```text
warmups/cora_gcn/data/cora.pkl
```

The dataset file is not tracked in Git. It contains:

| Field | Type | Description |
| --- | --- | --- |
| `x` | numpy array `(2708, 1433)` | node feature matrix |
| `y` | numpy array `(2708,)` | node labels, with test labels set to `-1` |
| `edge_index` | numpy array `(2, num_edges)` | graph edges |
| `train_mask` | numpy bool array `(2708,)` | training node mask |
| `val_mask` | numpy bool array `(2708,)` | validation node mask |
| `test_mask` | numpy bool array `(2708,)` | test node mask |
| `num_classes` | int | number of classes |
| `num_features` | int | feature dimension |

## Run

From the repository root:

```bash
source env.sh
python warmups/cora_gcn/gcn.py \
  --data-path warmups/cora_gcn/data/cora.pkl \
  --output warmups/cora_gcn/result.json \
  --seed 42 \
  --epochs 200
```

Useful options:

- `--cpu`: force CPU execution
- `--hidden-dim`: hidden feature size, default `256`
- `--dropout`: dropout rate, default `0.8`
- `--lr`: learning rate, default `0.01`
- `--weight-decay`: Adam weight decay, default `5e-4`

## Output

The script writes:

```text
warmups/cora_gcn/result.json
```

Package it for the platform with:

```bash
cd warmups/cora_gcn
zip ../../submissions/warmup1-result.zip result.json
```
