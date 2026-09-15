# Track 1 B-list Jittor reproduction

The package has exactly two public reproduction commands. Both accept only the
official `data_B.zip` and produce the recorded B-list result
`1.5240999401892983`.

## Public commands

### 1. Frozen final-layer reproduction

This is the compact final-layer route. It restores the packaged frozen base and
MF32 state, then builds the recorded result byte-for-byte.

```bash
python code/main.py verify --data /path/to/data_B.zip \
  --output /data1/b-verify --gpu 0
```

Expected output:

```text
/data1/b-verify/result.zip
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

### 2. Full-chain reproduction

This route executes the complete Dataset3/Dataset4 training, feature,
replay, graph, fusion, and serialization chain from `data_B.zip`, then
finalizes the recorded B-list state and writes the same exact result.

```bash
python code/main.py reproduce --data /path/to/data_B.zip \
  --output /data1/b-reproduce --gpu 0
```

Expected output:

```text
/data1/b-reproduce/result.zip
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

`/data1/b-reproduce/REPRODUCTION_RECEIPT.json` records the full-pipeline
`result.zip`, its serialized fresh frozen-base hash, the final frozen-base hash,
and the final result hash. This keeps the upstream execution and final result
bound in one receipt.

## Full-chain architecture

```text
data_B.zip
  -> C2 source/session training
  -> C3 multiscale gate
  -> C5 session ring
  -> C6 tie-group modelling
  -> RUC4 Set Transformer and session graph
  -> temporal, MF512, transition-MF, and pair-new experts
  -> replay caches, hierarchy/neighbor features, 75-feature meta ranker
  -> base-score result.zip
  -> q35/q7 frozen serialization
  -> recorded final state and MF32 rank construction
  -> result.zip
```

`code/pipeline/reproduce_full.py` drives the full route. Its upstream graph is:

```text
reproduce.py
  -> reproduce_third_1.py
  -> reproduce_ruc4.py
  -> reproduce_c6.py
  -> reproduce_c5.py
  -> reproduce_c3.py
  -> c2_source/reproduce_c2.py
```

Dataset3 uses source/session statistics, multiscale support, session-ring and
tie-group signals, and a three-seed candidate Set Transformer. Dataset4 uses
history/test-pool temporal experts, three 512-dimensional implicit-MF members,
transition-MF, a six-member pair-new Transformer, session-graph hard negatives,
RP3/RUC2/RUC3/RUC4 fusion, replay caches, hierarchy/neighbor features, and a
75-feature Jittor meta ranker.

The full route first writes a base-score `result.zip`; this is an intermediate
score archive, not the final submission. `pack_frozen_base.py` serializes it as
Dataset3 q35+LZMA and Dataset4 q7:

| Member | Encoding |
| --- | --- |
| `dataset3_q35_lzma` | signed fixed-point `1e10`, ZigZag, 35 bit-planes, LZMA |
| `dataset4_q7` | clipped q7 scores, little-endian row-major packing |

## Environment

Target environment: Ubuntu 22.04, NVIDIA RTX 4090, CUDA 12.4-compatible
driver/toolkit, Python 3.10, and Jittor 1.3.10.0.

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev \
  build-essential unzip

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3.10 -m venv /data1/sader-repro-py310
source /data1/sader-repro-py310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the shared preflight:

```bash
export ML_CACHE_ROOT=/data1/sader-runtime
source code/prepare_cuda_runtime.sh python
python code/check_environment.py
```

## Data boundary

All training and inference code uses the official archive only. It does not use
test-set ground truth, external datasets, external predictions, or leaked
labels.

## Package map

- `code/main.py`: the two public commands.
- `run_verify.sh`: frozen final-layer reproduction.
- `run_reproduce.sh`: full-chain reproduction.
- `code/pipeline/`: complete Dataset3/Dataset4 training graph and serializer.
- `code/build_submission.py`: deterministic final result construction.
- `code/assets/locked/`: retained final frozen base and MF32 state.
- `code/audit_package.py`: package-integrity verification.
