# Code map

`main.py` is the single Python entry point. It delegates to the shell
launchers at the package root so that cache setup, CUDA compatibility setup,
non-overwrite checks, and the original argument contracts remain identical.

## Commands

```bash
python code/main.py verify --data /path/to/data_A.zip --output /path/to/new_verify_output
python code/main.py infer --data /path/to/data_A.zip --output /path/to/new_output
python code/main.py train --data /path/to/data_A.zip --models /path/to/new_models --dataset all --cuda
python code/main.py fresh-infer --data /path/to/data_A.zip --models /path/to/models --output /path/to/new_output
python code/main.py raw --data /path/to/data_A.zip --models /path/to/new_models --output /path/to/new_output
```

`verify` is the required recorded-result path. Its execution graph is:

```text
main.py verify
  -> run_verify.sh
     -> MANIFEST.sha256 and audit_exact_package.py
     -> run_inference.sh
        -> build_community_residual_submission.py
           -> dataset1/d1_source_support_postprocess.py
           -> dataset2/d2_exact_group_postprocess.py
           -> Jittor BPR32 cosine residual
           -> result.zip and build_audit.json
```

The builder reads the official archive, the included frozen score artifact,
the included Jittor BPR32 checkpoint, and the locked Dataset1 configuration.
It rejects changed inputs and an output ZIP whose SHA-256 differs from the
recorded result. It does not open test labels. The launcher first checks the
pinned Python/Jittor/numerical dependency versions and a CUDA Jittor
arithmetic probe.

## Directory responsibilities

| Path | Responsibility |
| --- | --- |
| `build_community_residual_submission.py` | Exact submission construction, input hashes, Jittor cosine residual, canonical CSV and ZIP writing. |
| `dataset1/` | Source-support adjustment for Dataset1. |
| `dataset2/` | Exact candidate-group and community residual adjustment for Dataset2. |
| `raw_training/main.py` | Raw Jittor training and fresh-inference coordinator. |
| `raw_training/dataset1/` | Two-member temporal graph ranker training and scoring. |
| `raw_training/dataset2/` | Jittor VAE, RecVAE, BPR, pool, set, Transformer, and warm-residual components. |
| `raw_training/legacy_dataset2/` | Retained Jittor legacy ranker components and reconstruction helper. |
| `tools/audit_exact_package.py` | Static framework, artifact, data-boundary, and report checks. |

The raw commands train fresh models into the caller-selected model directory
and record manifests. They are provided as the complete training source;
their output is separately audited and is not asserted to be byte-identical
to the recorded historical result. The `verify` command is the exact
recorded-result reconstruction path.
