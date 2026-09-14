# Frozen base generation (official-data full pipeline)

This directory reconstructs `models/frozen_base.ckpt` from the official
`data_B.zip`. The published package ships the frozen base and reranks it with
an MF32 residual in `code/build_submission.py`; the code here is the upstream
half that produces the base itself, closing the `official data -> base scores
-> frozen_base.ckpt -> submission` chain.

## Flow

```text
official data_B.zip
  -> reproduce_third_1.py    train every D3/D4 component from scratch -> result.zip
  -> pack_frozen_base.py     score matrices -> frozen_base.ckpt (D3 q35+lzma, D4 q7)
  -> code/build_submission.py  frozen base + MF32 residual -> result.zip
```

`reproduce_third_1.py` drives the full training pipeline: it rebuilds the ruc4
candidate base (via the nested `reproduce_ruc4 -> c6 -> c5 -> c3 -> c2_source`
chain), trains the temporal, implicit-MF, transition-MF and pair-new rankers,
fits the multi-model ensemble, builds the hierarchy/neighbour meta features,
trains the meta ranker with Jittor, and writes the base score matrices to
`result.zip`. No pretrained weights, external data, or answer files are used.

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
python generation/generate_frozen_base.py --data /path/to/data_B.zip \
  --work-dir /data1/b-frozen-smoke --quick

# prove the codec is the exact inverse of the consumer (no GPU/data needed)
python generation/pack_frozen_base.py --self-test
```

The work directory must not already exist. The generated base and a
`REPRODUCTION_RECEIPT.json` with the actual hashes are written there. Feed the
base into the existing inference path:

```bash
bash run_inference.sh /path/to/data_B.zip /data1/b-output 0   # after copying the
                                                              # base to models/
```

## Approximate reconstruction

The regenerated frozen base is **not** guaranteed to match the recorded locked
base byte-for-byte, for two reasons:

1. **Missing historical checkpoints.** The recorded base drew on stacker and
   pair-new checkpoints that are not shipped; the pipeline retrains equivalents
   rather than restoring the exact historical arrays, so the score matrices
   differ slightly.
2. **Jittor operator perturbation.** Jittor's CUDA kernels are not bit-identical
   across machines and driver/toolkit versions, so low-order bits of the scores
   drift from run to run. Rank order is stable; exact bytes are not.

Byte-for-byte reproduction of the recorded online submission therefore remains
the job of `run_verify.sh`, which consumes the retained locked base. This
directory documents and reproduces the *method* that produces such a base.

## Environment

Same target as the package: Ubuntu 22.04, Python 3.10, CUDA 12.4-compatible
toolkit, and Jittor. The pipeline additionally uses `numba` (already pinned in
`requirements.txt`). Run `code/prepare_cuda_runtime.sh` and
`code/check_environment.py` first, exactly as the other launchers do;
`run_generate_base.sh` performs both preflight steps automatically.
