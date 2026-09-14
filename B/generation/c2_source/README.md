# c2 online 1.2133719042789612

This source-only directory reproduces the c2 algorithm from the official
`data_B.zip`. It contains no official data, submission ZIP, cache, model, or
checkpoint.

Online evidence:

- score: `1.2133719042789612`
- submitted ZIP SHA-256: `7f223dc558bcab9f8c4dfb2e04ad151a6ad65d06e4e1ef0026910d8fa1abed6e`
- previous v26 score: `1.2019175449740571`
- online improvement: `+0.011454359304904127`

The c2 change is D3-only. It retains the v26 exact-time cross-source residual
(`weight=0.10`) and adds same-source candidate recurrence within `+/-300s`,
excluding exact-time support (`weight=0.05`). D4 is byte-identical to the
online v26/c1 submission. The strict replay gate measured:

- validation: `+0.0045717173 +/- 0.0003459241` MRR
- confirmation: `+0.0045562286 +/- 0.0002891328` MRR
- confirmation negative-row rate: `0.0000771724`

## Reproduce

Use Python 3.10.20 and the exact versions in `requirements.txt`. The online
environment used CUDA 12.0.76 and GCC 11.4 for Jittor compilation:

```bash
python reproduce_c2.py \
  --data /path/to/data_B.zip \
  --work-dir /new/path/c2_run \
  --gpu 0
```

The input SHA-256 must be
`ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2`.
All generated caches, checkpoints, reports, and the final
`b_rank_d34_c2_source_session.zip` stay under `--work-dir`.
The runner also isolates `HOME` and `JITTOR_HOME` under that directory so an
incompatible global Jittor core cache cannot leak into the run.

For a small linkage check only:

```bash
python reproduce_c2.py --data /path/to/data_B.zip --work-dir /new/path/smoke --gpu 0 --quick
```

`--quick` writes a `SMOKE_ONLY` receipt and must not be submitted. A fresh
training run rebuilds the algorithm and policy from the official archive.
The runnable data-to-submission implementation is the `reproduce_c2.py`
pipeline; no frozen evidence file is bundled or required.
