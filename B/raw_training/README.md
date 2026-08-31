# B-list official-data training

This directory contains the B-list training coordinator and base generator.
The workflow reads the competition's official `data_B.zip`, trains the Jittor
MF32 model, constructs the candidate base, and produces the states consumed by
the shared inference builder.

```text
official data_B.zip
  -> code/train_model.py
  -> d4_implicit_mf32.npz
  -> raw_training/build_base.py
  -> base_result.zip
  -> run_fresh_inference.sh
  -> result.zip
```

## Commands

```bash
bash run_train.sh /path/to/data_B.zip /data1/b-models 0
bash run_fresh_inference.sh /path/to/data_B.zip \
  /data1/b-models /data1/b-output 0
```

`run_train.sh` writes `d4_implicit_mf32.npz`, `TRAINING_RECEIPT.json`,
`base_result.zip`, `BASE_RECEIPT.json`, and `RAW_TRAINING_RECEIPT.json`.
`run_fresh_inference.sh` validates those states, runs the shared Jittor model
and serializer, and writes `result.zip` plus `FRESH_RUN_RECEIPT.json`.

For byte-for-byte reproduction of the highest online submission, the reviewer
supplies the official `data_B.zip` and runs:

```bash
bash run_verify.sh /path/to/data_B.zip /data1/b-byte-exact 0
```

That authoritative command validates the package assets and accepts only the
recorded result SHA-256.

All code in this directory follows the official-data interface. It contains no
use or leakage of test-set ground truth, leaked labels, external predictions,
or non-official datasets.
