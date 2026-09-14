# Frozen base generation (official-data full pipeline)

This directory reconstructs `models/frozen_base.ckpt` from the official
`data_B.zip`. The published package ships the frozen base and reranks it with
an MF32 residual in `code/build_submission.py`; the code here is the upstream
half that produces the base itself, closing the `official data -> base scores
-> frozen_base.ckpt -> submission` chain.

## Flow

```text
official data_B.zip
  -> reproduce.py            end-to-end D3/D4 training from scratch -> result.zip
  -> pack_frozen_base.py     score matrices -> frozen_base.ckpt (D3 q35+lzma, D4 q7)
  -> code/build_submission.py  frozen base + MF32 residual -> result.zip
```

`reproduce.py` is the self-contained, weight-free end-to-end entry (the
`d3d4_official_reproduce_v2` pipeline). It audits the source tree, checks the
Jittor CUDA linkage, then runs `reproduce_third_1.py`, which rebuilds the ruc4
candidate base (via the nested `reproduce_ruc4 -> c6 -> c5 -> c3 -> c2_source`
chain), trains the temporal, transition-MF, pair-new and 512-dimensional
implicit-MF experts, fits the multi-model ensemble, builds the
hierarchy/neighbour meta features, trains the Jittor candidate-set fusion
model, and writes the base score matrices to `result.zip`. Supervision is only
`dataset3/train.csv` and `dataset4/train.csv`; no pretrained weights, external
data, answer files, or the frozen base itself are read.

`pack_frozen_base.py` is the encoder that was missing from the package. It is
the exact inverse of the decoders in `code/build_submission.py`:

| Member | Encoding | Inverse of |
| --- | --- | --- |
| `dataset3_q35_lzma` | zig-zag int64, 35 bit-planes, LZMA, scale `1e10` | `decode_dataset3_q35` |
| `dataset4_q7` | 7-bit row-major little-endian behind a fixed magic | `binary_q7_chunks` |

## Commands

```bash
# via the unified entry point (output path is the work directory)
python code/main.py generate-base --data /path/to/data_B.zip \
  --output /data1/b-frozen-base --gpu 0

# or directly
bash run_generate_base.sh /path/to/data_B.zip /data1/b-frozen-base 0

# compile-only smoke test (no base produced)
python code/pipeline/generate_frozen_base.py --data /path/to/data_B.zip \
  --work-dir /data1/b-frozen-smoke --quick

```

The work directory must not already exist. The generated base and a
`REPRODUCTION_RECEIPT.json` with the actual hashes are written there. Feed the
base into the existing inference path:

```bash
bash run_inference.sh /path/to/data_B.zip /data1/b-output 0   # after copying the
                                                              # base to models/
```

## Scope

This directory supplies the frozen-base generation chain: it trains the
Dataset3/Dataset4 components from the official data and packs the resulting
score matrices into `frozen_base.ckpt`, which `code/build_submission.py` then
reranks to reproduce the recorded top submission. It is the upstream generation
half only; supervision is the official training data alone, and no answer
files, external data, or the frozen base itself are read.

## Environment

Same target as the package: Ubuntu 22.04, Python 3.10, CUDA 12.4-compatible
toolkit, and Jittor. The pipeline additionally uses `numba` (already pinned in
`requirements.txt`). Run `code/prepare_cuda_runtime.sh` and
`code/check_environment.py` first, exactly as the other launchers do;
`run_generate_base.sh` performs both preflight steps automatically.
