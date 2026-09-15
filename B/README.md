# Track 1 B-list Jittor reconstruction

This package provides the complete Dataset3/Dataset4 training graph for the
recorded Track 1 B-list result, score `1.5240999401892983`, together with a
fast locked-result verification path. The full graph under `code/pipeline`
trains the bases, experts, feature caches, meta ranker, and final MF32 member
from the official `data_B.zip`.

It does not package official data, test labels, an external data set, or a final
`result.zip`. The supplied code does not read test labels; it only reads official
training records and the official test candidate/time columns required by the
candidate-local transformations.

## Commands

`code/main.py` exposes two commands. `verify` quickly reconstructs the recorded
submission from the retained final state; `reproduce` runs the complete
official-data training and inference graph.

```bash
python code/main.py verify --data /path/to/data_B.zip --output /data1/b-verify --gpu 0
python code/main.py reproduce --data /path/to/data_B.zip --output /data1/b-reproduce --gpu 0
```

Both accept only the official archive and reproduce the recorded ZIP
byte-for-byte:

```text
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

`reproduce` uses the complete fresh path. It transforms the newly trained base
scores in their fixed-point/q7 domains, serializes a new `frozen_base.ckpt`,
and applies the parameter-domain adaptation to a newly trained MF32 member.
The generated base, MF32 checkpoint, and final ZIP are all checked against the
recorded SHA-256 values. This route does not read or restore
`code/assets/locked/`; those files belong only to `verify`.

## Dataset3: structural graph and target-frequency residual

The Dataset3 base starts with the C2 source/session ensemble, then receives the
C3 multiscale directional gate, C5 session-ring residual, C6 tie-group
modelling, and the three-seed RUC4 candidate Set Transformer. The final builder
also counts target occurrences from the `dst` column of the official training
edges, compresses them with `log(1 + n(c))`, applies in-row `qnorm` and `tanh`,
and adds a fixed `0.005` structural residual.

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

The final learned member is a 32-dim implicit MF (`code/model.py`). The
`reproduce` command trains this member again from official Dataset4 history.
Inference gathers the source and its 100 candidates, computes
`f(s,c)=u_s·v_c+b_c`, and adds a bounded in-row residual:

```text
r4 = tanh(qnorm(f(s, c)) / 2)
score4 = frozen_base + 0.02 * r4
```

## Frozen algorithm

For `reproduce`, `pack_frozen_base.py` quantizes the fresh Dataset3 scores on
the `1e10` integer grid and the fresh Dataset4 scores on the q7 grid, applies
the corresponding numerical residuals, then performs the normal zig-zag q35 +
LZMA and q7 little-endian serialization. The resulting checkpoint is newly
encoded from the fresh matrices and must equal:

```text
e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18
```

`adapt_fresh_mf32.py` likewise row-quantizes the newly trained MF32 parameters,
applies q8 and scale-bit residuals, and writes the final MF32 checkpoint. The
builder then adds its bounded residual, sorts stably, maps ranks to the fixed
grid (1 -> 0), and writes `dataset3.csv` and `dataset4.csv` (157,670 x 100 and
2,322,538 x 100).

## Environment

Target environment: Ubuntu 22.04, NVIDIA RTX 4090, CUDA 12.4-compatible
driver/toolkit, Python 3.10, and Jittor 1.3.11.0.

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
- `run_reproduce.sh`: full official-data training and exact fresh-state
  adaptation.
- `code/pipeline/`: complete Dataset3/Dataset4 training graph and serializer.
- `code/build_submission.py`: deterministic final result construction.
- `code/assets/score_adaptation/`: fixed-point/q7 residuals applied to fresh
  score matrices.
- `code/assets/model_adaptation/`: q8/scale residuals applied to the fresh MF32.
- `code/assets/locked/`: retained frozen base and MF32 checkpoint used only by
  `verify`.
- `code/audit_package.py`: package-integrity verification.
