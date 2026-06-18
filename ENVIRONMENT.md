# Jittor Competition Environment

This workspace is configured for the Jittor AI competition on a CPU-first path.

## Activate

```bash
source env.sh
```

Then run:

```bash
python verify_env.py
```

Expected highlights:

- Python path: `/tmp/ml_venv/bin/python`
- `jittor 1.3.11.0`
- `has_cuda False`
- `sum 6`
- `gcn_output_shape (4, 2)`
- `environment ok`

## Why `env.sh` Uses `/tmp`

The project directory is named `M L`, and the space breaks a few shell commands
inside Jittor/JittorGeometric. `env.sh` creates a no-space virtualenv symlink at
`/tmp/ml_venv` and stores Jittor caches under `/tmp/ml_jittor_home`.

It also sets:

- `use_mpi=0`, because OpenMPI socket initialization is blocked in this sandbox.
- `nvcc_path=""`, because GPU access is not usable here and Jittor would
  otherwise try to download a large CUDA package.

## Installed Core Packages

- `jittor==1.3.11.0`
- `jittor_geometric==2.0.0`
- `numpy`, `scipy`, `scikit-learn`, `pandas`, `networkx`, `tqdm`

`jittor_geometric` was installed from:

```text
https://github.com/AlgRUC/JittorGeometric.git
commit ff7d8ffac7bf3d95cc1962e091c52dc5737492d4
```

## Local Compatibility Patches

The installed JittorGeometric package was patched inside `.venv` so CPU imports
work:

- `ops/spmmcsr.py`: removed import-time `jt.flags.use_cuda=1`; CUDA is now
  required only when `SpmmCsr` is actually called.
- `ops/spmmcoo.py`: same change for `SpmmCoo`.
- `ops/__init__.py`: made `getweight`, `sampleprocessing`, and `gpuinitco`
  lazy imports, because they compile optional CUDA/sample operators during
  import.

These patches let the warm-up GCN path use CPU `aggregateWithWeight`.

## GPU Later

For real leaderboard training, especially formal tracks, use a machine with
working NVIDIA GPU, CUDA/NVCC, and `nvidia-smi`. On that machine, remove or
override:

```bash
unset nvcc_path
```

Then run Jittor's CUDA test:

```bash
python -m jittor.test.test_cuda
```
