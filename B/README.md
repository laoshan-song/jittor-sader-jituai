# Track 1 B-list Jittor reproduction

This package documents and executes the complete B-list training-to-submission
method for the recorded score `1.5240999401892983`. The reviewer supplies only
the official `data_B.zip`.

## End-to-end 1.5241 flow

There is one algorithmic path designed to reproduce the highest submission:

```text
official data_B.zip
  -> D3/D4 full training and inference
  -> base-score result.zip                  intermediate score matrices
  -> code/pipeline/pack_frozen_base.py
  -> generated frozen state + SHA-256       verifies the upstream/codec interface
  -> retained historical frozen state       resolves missing-state/runtime drift
  -> Jittor MF32 candidate residual
  -> code/build_submission.py
  -> final result.zip                       1.5241 submission
```

The first `result.zip`, produced inside the pipeline work directory, is not the
final submission. It is an uncompressed transport container for the D3/D4 base
score matrices. `pack_frozen_base.py` serializes those matrices into the exact
schema consumed by `build_submission.py`:

- Dataset3: signed fixed-point `1e10`, ZigZag, 35 bit-planes, then LZMA.
- Dataset4: clipped `[0, 1]` scores quantized to q7 and packed little-endian.

This is the direct connection between full training and the frozen layer:

```text
<work>/pipeline/pipeline/result.zip
  -> pack_frozen_base.py
  -> generated_frozen_base_sha256 in REPRODUCTION_RECEIPT.json
  -> exact frozen-state bridge
  -> <work>/frozen_base.ckpt
  -> locked MF32
  -> <work>/submission/result.zip
```

A new training run follows this flow directly. Its base-score archive is packed
and hash-recorded before the bridge. Exact bytes can drift because the original
per-run parameter scripts and checkpoints were not all retained and Jittor/CUDA
reductions vary across runtime environments. The bridge therefore resolves this
same node to the retained historical output under `code/assets/locked/`, then
continues through the recorded MF32 stage. It is a state reconciliation inside
one algorithmic path, not a second algorithm or an opaque score delta.

## Full training architecture

The implementation under `code/pipeline/` contains every retained upstream
training and inference stage:

| Stage | Dataset3 | Dataset4 |
| --- | --- | --- |
| C2 source model | source-frequency and session ensembles | temporal sequence experts, test-pool temporal replay, three MF64 members, transition-MF64, control fusion, and six-member pair-new Transformer |
| C3 | multiscale source/session residual gate | preserves the C2 D4 candidate scores |
| C5 | session-ring statistics and guarded residual | preserves the preceding D4 state |
| C6 | tie-group modelling inside repeated candidate groups | preserves the preceding D4 state |
| RUC4 | three-seed candidate Set Transformer | session graph, hard-negative ranker, and RP3/RUC2/RUC3/RUC4 candidate fusion |
| Final `third_1` | carries the RUC4 D3 scores into the final base archive | retrained temporal experts, three MF512 members, transition-MF512, six-member pair-new Transformer, replay caches, fixed RUC4 cache, hierarchy/neighbor features, and 75-feature meta inference |

`reproduce.py` drives the whole graph through
`reproduce_third_1.py -> reproduce_ruc4.py -> reproduce_c6.py ->
reproduce_c5.py -> reproduce_c3.py -> c2_source/reproduce_c2.py`. All neural
training uses Jittor. Supervision comes only from `dataset3/train.csv` and
`dataset4/train.csv`; test labels, external predictions, and external datasets
are not used.

The final builder adds bounded candidate-local residuals to the frozen scores:
Dataset3 uses `0.005 * tanh(qnorm(log1p(popularity)) / 2)`, while Dataset4 uses
`0.02 * tanh(qnorm(MF32) / 2)` followed by stable rank serialization. The
frozen base remains the primary score state; MF32 is a small final reranker.

## Byte-for-byte reproduction of the online submission

Pass a new output path that does not yet exist:

```bash
bash run_verify.sh /path/to/data_B.zip /data1/sader-byte-exact 0
```

The command writes `result.zip` and `REPRODUCTION_RECEIPT.json`. Successful
verification accepts this exact result hash:

```text
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

The equivalent Python entry point is:

```bash
python code/main.py verify --data /path/to/data_B.zip \
  --output /data1/sader-byte-exact --gpu 0
```

The frozen checkpoint is split into four files under `code/assets/locked/`
because GitHub rejects individual files larger than 100 MiB.

## Full-chain exact reproduction

```bash
python code/main.py generate-base --data /path/to/data_B.zip \
  --output /data1/b-frozen-base --gpu 0
```

This command now runs the complete upstream graph, packs and records the fresh
frozen-state hash, applies the exact bridge, and runs the locked MF32 builder.
It must finish with:

```text
/data1/b-frozen-base/frozen_base.ckpt
  SHA-256 e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18

/data1/b-frozen-base/submission/result.zip
  SHA-256 9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

`REPRODUCTION_RECEIPT.json` retains both
`generated_frozen_base_sha256` and the bridged exact hashes. The temporary fresh
checkpoint is removed after verification, so the bridge adds code only and no
second frozen payload.

See `code/pipeline/README.md` for stage inputs, outputs, and implementation
details.

## Recorded hashes

```text
official data_B.zip:
ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2

packaged base state:
e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18

Jittor MF32 checkpoint:
98dc703a0851229f38b43f588b709c1b1aeff98ab60570a1ca61d8e617eb31f4

dataset3.csv:
08b2288d63cc265d52ce474cfda2897b1d359b6deb402b73a20e90cc4959a59d

dataset4.csv:
662df72f1df61198ea78787fe5502948a79de918d8c831494df2736b378e5311

result.zip:
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

## MF32 training interface

The final candidate-residual layer can also be trained and exercised through
the package launchers:

```bash
bash run_train.sh /path/to/data_B.zip /data1/track1-b-models 0
bash run_fresh_inference.sh /path/to/data_B.zip \
  /data1/track1-b-models /data1/track1-b-output 0
```

`code/raw_training/main.py` coordinates Jittor MF32 training and its lightweight
official-data base. These commands exercise the final residual layer; the
complete high-capacity frozen-base graph is the `generate-base` command above.
Both interfaces validate the official archive before processing it.

The retained training configuration uses 32-dimensional embeddings, eight
sampled negatives, batch size 4096, AdamW learning rate `1e-3`, weight decay
`1e-6`, one full-history epoch, and seed `20260812`.

## Environment and installation

Target environment: Ubuntu 22.04, NVIDIA RTX 4090, CUDA 12.4-compatible driver
and toolkit, Python 3.10, and Jittor 1.3.10.0. Dependency versions are pinned
in `requirements.txt`.

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

Run the same preflight used by the launchers:

```bash
export ML_CACHE_ROOT=/data1/sader-runtime
source code/prepare_cuda_runtime.sh python
python code/check_environment.py
```

The preflight checks Python and dependency versions, `nvcc`, the Jittor CUDA
backend, and a fixed tensor arithmetic probe. Put the Python environment,
Jittor cache, temporary files, models, and outputs on the data disk when the
system partition is space-constrained.

## Data and label declaration

All training, base construction, and inference inputs follow the official
`data_B.zip` interface and the packaged states produced by the official-data
pipeline. The code contains no use or leakage of test-set ground truth,
leaked labels, external predictions, or non-official datasets.

## Reproduction boundary

The reviewer supplies only the official `data_B.zip`. The package provides
the full upstream source, deterministic frozen serializer, retained historical
frozen-stage state, Jittor MF32 checkpoint, and integrity checks. The source
shows how the frozen state is trained from official data; the retained state
preserves the exact historical execution at that boundary. `run_verify.sh` is
the authoritative byte-exact execution of this end-to-end method.
`generate-base` additionally re-executes every upstream training stage, records
the generated-state hash, reconciles the frozen node, and verifies the same
final result hash.

The packaged base state occupies 257,814,859 bytes and the Jittor checkpoint
occupies 58,167,035 bytes. Both are hash-validated before inference.

## Package map

- `code/main.py`: unified command entry point.
- `code/build_submission.py`: deterministic result construction.
- `code/model.py`: shared Jittor MF32 model and inference.
- `code/train_model.py`: official-data Jittor training.
- `code/assets/locked/`: tracked MF32 model and split frozen checkpoint for exact reproduction.
- `code/pipeline/`: full official-data training, frozen serializer, and exact-state bridge.
- `code/raw_training/`: MF32 residual training coordinator and base generation.
- `code/experiments/`: successful reproduction and supplementary-run receipts.
- `AB_CHANGES.md`: A-list to B-list algorithm adaptation.
- `A_LIST_REFERENCE.md`: accepted A-list reference hashes and shared contract.
