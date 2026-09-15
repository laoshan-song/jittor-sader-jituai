# Track 1 B-list Jittor reconstruction

This package provides the source code and the locked reconstruction path for
the recorded Track 1 B-list result, score `1.5240999401892983`. The locked path
uses the official `data_B.zip`, the retained frozen base artifact, a Jittor
MF32 checkpoint, and fixed postprocessing to regenerate the submission ZIP
byte-for-byte. The full Dataset3/Dataset4 training graph is under `code/pipeline`.

It does not package official data, test labels, an external data set, or a final
`result.zip`. The supplied code does not read test labels; it only reads official
training records and the official test candidate/time columns required by the
locked transformations.

## Commands

`code/main.py` exposes two commands. `verify` is the recommended
recorded-result reconstruction; `reproduce` runs the full official-data graph.

```bash
python code/main.py verify --data /path/to/data_B.zip --output /data1/b-verify --gpu 0
python code/main.py reproduce --data /path/to/data_B.zip --output /data1/b-reproduce --gpu 0
```

Both accept only the official archive and both produce the recorded result ZIP:

```text
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

`verify` restores the retained frozen base and MF32 checkpoint and rebuilds the
result. `reproduce` retrains the full Dataset3/Dataset4 graph, then aligns to
the retained final state and rebuilds the result. As on the A list, a
from-scratch run is not guaranteed to be bit-identical to the historical
snapshot; the retained final state and the fixed result hash define the
exact-reproduction contract.

## Dataset3: target-frequency structural residual

Dataset3 (built by `code/build_submission.py`) is the statistics member. It
counts target occurrences from the `dst` column of the official training edges,
compresses with `log(1 + n(c))`, maps candidate ids through the sorted
vocabulary, applies in-row `qnorm` and `tanh`, and adds a fixed `0.005` residual
to the frozen base score. It is a low-magnitude structural prior that only
separates near-tied candidates.

## Dataset4: 512-dim expert graph and 32-dim MF member

The Dataset4 base is trained by the graph under `code/pipeline` from
`data_B.zip` only (`reproduce.py -> reproduce_third_1.py -> reproduce_ruc4.py ->
reproduce_c6.py -> reproduce_c5.py -> reproduce_c3.py ->
c2_source/reproduce_c2.py`):

- **C2 source/session base**: source/session frequency base with temporal
  sequence experts, test-pool replay, three 512-dim implicit-MF members,
  transition-MF, and a six-member pair-new Transformer.
- **C3**: multiscale source/session directional-support gate.
- **C5**: session-ring support and guarded residual.
- **C6**: tie-group modelling inside repeated candidate groups.
- **RUC4**: three-seed candidate Set Transformer with session graph, hard
  negatives, and RP3/RUC2/RUC3/RUC4 fusion.
- **third_1**: replay caches, hierarchy/neighbor features, and a 75-feature
  Jittor meta ranker that produces the base score matrices.

The final learned member is a 32-dim implicit MF (`code/model.py`). Inference
gathers the source and its 100 candidates, computes `f(s,c)=u_s·v_c+b_c`, and
adds a bounded in-row residual:

```text
r4 = tanh(qnorm(f(s, c)) / 2)
score4 = frozen_base + 0.02 * r4
```

## Frozen algorithm

The base score matrices are serialized with `pack_frozen_base.py` (Dataset3
zig-zag `1e10` q35 + LZMA, Dataset4 q7 little-endian packing) into the frozen
base. `code/build_submission.py` then adds the bounded candidate-local
residuals above, sorts stably, maps ranks to the fixed grid (1 -> 0), and writes
`dataset3.csv` and `dataset4.csv` (157,670 x 100 and 2,322,538 x 100). All
constants and input hashes are pinned in the builder; output acceptance is by
the fixed `result.zip` SHA-256, not by any local or online-score proxy.

## Environment

Target environment: Ubuntu 22.04, NVIDIA RTX 4090, CUDA 12.4-compatible
driver/toolkit, Python 3.10, and Jittor 1.3.10.0.

```bash
python3.10 -m venv /data1/sader-repro-py310
source /data1/sader-repro-py310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
- `run_verify.sh`: locked reconstruction from the retained final state.
- `run_reproduce.sh`: full official-data graph, then aligned reconstruction.
- `code/pipeline/`: complete Dataset3/Dataset4 training graph and serializer.
- `code/build_submission.py`: deterministic final result construction.
- `code/assets/locked/`: retained frozen base and MF32 checkpoint.
- `code/audit_package.py`: package-integrity verification.
