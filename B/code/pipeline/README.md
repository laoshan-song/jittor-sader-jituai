# Full training and frozen-state generation

This directory contains the complete upstream training portion of the B-list
`1.5241` method. It trains the Dataset3 and Dataset4 stack from the official
`data_B.zip`, emits base score matrices, and serializes those matrices as the
`frozen_base.ckpt` consumed by the final MF32 reranker.

## Flow

```text
official data_B.zip
  -> reproduce.py
  -> reproduce_third_1.py
  -> nested D3/D4 training, replay, graph, and meta-model stages
  -> <work>/pipeline/pipeline/result.zip        base-score result, not final submission
  -> pack_frozen_base.py
  -> <work>/frozen_base.ckpt                    frozen intermediate state
  -> code/build_submission.py + MF32
  -> final result.zip
```

`reproduce.py` is the self-contained, weight-free entry. It audits the source
tree, checks Jittor CUDA linkage, and invokes the complete training graph.

## Training stages

| Order | Driver/output | Retained method |
| --- | --- | --- |
| 1 | `c2_source/reproduce_c2.py` | D3 source-frequency and session ensemble; D4 temporal experts, test-pool replay, implicit MF, transition MF, control fusion, and pair-new Transformer |
| 2 | `reproduce_c3.py` | D3 multiscale source/session directional support and candidate-local gate |
| 3 | `reproduce_c5.py` | D3 session-ring support and guarded residual |
| 4 | `reproduce_c6.py` | D3 tie-group modelling inside repeated candidate groups |
| 5 | `reproduce_ruc4.py` | Three-seed D3 candidate Set Transformer; D4 session graph, hard-negative ranker, and RP3/RUC2/RUC3/RUC4 fusion |
| 6 | `reproduce_third_1.py` | History and test-pool temporal experts, three MF512 members, transition-MF512, six-member pair-new Transformer, replay caches, fixed RUC4 cache, hierarchy and neighbour features, and a 75-feature Jittor meta ranker |
| 7 | `pack_frozen_base.py` | Deterministic D3 q35+LZMA and D4 q7 serialization |

The nested order is:

```text
reproduce_third_1.py
  -> reproduce_ruc4.py
  -> reproduce_c6.py
  -> reproduce_c5.py
  -> reproduce_c3.py
  -> c2_source/reproduce_c2.py
```

The output of stage 6 is named `result.zip` because pipeline stages use the
competition ZIP schema to transport score matrices. At this point it is a
base-score archive, not the final submission. Stage 7 reads `dataset3.csv` and
`dataset4.csv` and writes the frozen checkpoint. The final submission is
created only after `code/build_submission.py` adds the MF32 candidate residual.

Supervision is only `dataset3/train.csv` and `dataset4/train.csv`; no test
labels, external data, answer files, or tracked frozen scores are read by the
upstream training run.

## Frozen format and historical state

`pack_frozen_base.py` is the exact inverse of the decoders in
`code/build_submission.py`:

| Member | Encoding | Inverse of |
| --- | --- | --- |
| `dataset3_q35_lzma` | ZigZag int64, 35 bit-planes, LZMA, scale `1e10` | `decode_dataset3_q35` |
| `dataset4_q7` | 7-bit row-major little-endian behind a fixed magic | `binary_q7_chunks` |

This codec is the direct interface between full training and the final
reranker. Exact frozen bytes can still drift because all original parameter
scripts, checkpoints, and runtime numerical states were not retained. The
split checkpoint under `code/assets/locked/` preserves the historical output at
this same stage, removing that training/runtime disturbance when
`run_verify.sh` reproduces the final online ZIP byte-for-byte.

## Commands

```bash
# complete upstream training and frozen-state generation
python code/main.py generate-base --data /path/to/data_B.zip \
  --output /data1/b-frozen-base --gpu 0

# equivalent direct launcher
bash run_generate_base.sh /path/to/data_B.zip /data1/b-frozen-base 0

# compile-only smoke test; no base is produced
python code/pipeline/generate_frozen_base.py --data /path/to/data_B.zip \
  --work-dir /data1/b-frozen-smoke --quick
```

The work directory must not already exist. The generated base and a
`REPRODUCTION_RECEIPT.json` with actual hashes are written there. Continue the
generated state through the same MF32 builder:

```bash
python code/train_model.py --data /path/to/data_B.zip \
  --output-dir /data1/b-fresh-mf32
python code/build_submission.py --data /path/to/data_B.zip \
  --base /data1/b-frozen-base/frozen_base.ckpt \
  --checkpoint /data1/b-fresh-mf32/d4_implicit_mf32.npz \
  --output-dir /data1/b-fresh-result --unlocked
```

## Scope

This directory supplies the single algorithmic chain through
`frozen_base.ckpt`; `code/build_submission.py` supplies the final MF32 residual
and output serialization. The generated and retained historical checkpoints
share the same downstream contract. The generated state directly re-executes
the upstream method, while the retained state stabilizes the same node for
byte-level verification.

## Environment

Same target as the package: Ubuntu 22.04, Python 3.10, CUDA 12.4-compatible
toolkit, and Jittor. The pipeline additionally uses `numba` (already pinned in
`requirements.txt`). Run `code/prepare_cuda_runtime.sh` and
`code/check_environment.py` first; `run_generate_base.sh` performs both
preflight steps automatically.
