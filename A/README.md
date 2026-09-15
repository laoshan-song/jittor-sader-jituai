# Track 1 A-list exact inference reconstruction

This package provides the source code and the locked reconstruction path for
the recorded Track 1 A-list result, score `1.521072794155721`. The locked path
uses the official `data_A.zip`, the retained score-space base artifact, a
Jittor BPR32 checkpoint, and fixed postprocessing to regenerate the submission
ZIP byte-for-byte. The raw Jittor training source is under `code/raw_training`.

It does not package official data, test labels, an external data set, a final
`result.zip`, or a non-Jittor deep-learning framework. The supplied code does
not read test labels; it only reads official training records and the official
test candidate/time columns required by the locked transformations.

## Package layout

- `code/main.py`: single Python entry point. `verify` is the recommended
  recorded-result reconstruction command; see `code/README.md` for the
  command map and actual call graph.
- `code/build_community_residual_submission.py`: exact result builder.
- `code/dataset1/`: Dataset1 source-support postprocessor.
- `code/dataset2/`: Dataset2 exact-group postprocessor.
- `code/tools/audit_exact_package.py`: package, framework, hash and data
  boundary audit.
- `code/tools/prepare_cuda_runtime.sh`: creates a caller-local CUDA/cuDNN
  wrapper only when the system CUDA SDK lacks cuDNN development files.
- `code/raw_training/main.py`: actual raw training/inference coordinator.
- `code/raw_training/dataset1/`: two-member Jittor Dataset1 graph ranker.
- `code/raw_training/dataset2/`: Jittor VAE/RecVAE/BPR, pool, set,
  Transformer and warm-residual components.
- `code/raw_training/legacy_dataset2/`: Jittor legacy ranking components.
- `models/model_bpr32_prod_community_jittor.npz`: retained Jittor BPR32
  checkpoint. Jittor performs user-vector normalization and cosine scoring.
- `base_result.zip`: retained pre-postprocessing score artifact.
- `experiments/d1_source_support_locked.json`: locked Dataset1 rule.
- `run_inference.sh` / `run_verify.sh`: non-overwriting locked reconstruction
  entry points; `run_verify.sh` also runs manifest and static audits.
- `run_train.sh` / `run_fresh_inference.sh` / `run_raw.sh`: raw Jittor
  training, fresh inference, and the combined training/inference protocol.

## Environment

The supported target is Ubuntu 22.04, Python 3.10, Jittor 1.3.10.0 and a
CUDA 12.4-compatible NVIDIA runtime. Create an isolated environment on the
data disk and install the pinned requirements:

```bash
python3.10 -m venv /data1/contest1-repro-env
source /data1/contest1-repro-env/bin/activate
python -m pip install -r requirements.txt
export JITTOR_HOME=/data1/contest1-jittor-cache
```

Pandas and Numba are used only by the raw data processing/training source.
The declared compatibility environment additionally includes
`jittor-geometric>=0.1.0` and `scikit-learn==1.5.2`; neither package is
imported by the locked reconstruction or by the retained model code. No
PyTorch, TensorFlow, Paddle or external data set is used. If the system CUDA
SDK does not expose `cudnn.h`, the included launcher uses the installed
`nvidia-cudnn-cu12` package to create a local wrapper under `ML_CACHE_ROOT`;
it never modifies system CUDA. The supported execution path requires a CUDA
12.4-compatible NVIDIA runtime, `nvcc`, and a CUDA-capable Jittor 1.3.10.0.

## Submission-PDF tooling

The supplied document is generated from `code/tools/submission_report.html`
with `weasyprint==66.0` and `PyMuPDF==1.25.5`; both packages are pinned in
`requirements.txt`. Display equations retain their original LaTeX source in
HTML (`data-latex`) and are rendered as larger vector SVG equations during PDF
generation. The LaTeX source metadata is hidden from the visual PDF, so each
equation appears only once in typeset form. The fixed SVG renderer covers the
displayed softmax, residual-fusion, and qnorm equations, so the PDF does not
depend on a browser MathML implementation. On Ubuntu 22.04, install the
standard text/layout runtime once before generating the PDF:

```bash
sudo apt-get update
sudo apt-get install -y libcairo2 libpango-1.0-0 libpangoft2-1.0-0 \
  libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info fonts-noto-cjk
python -m pip install -r requirements.txt
python code/tools/make_submission_pdf.py --replace
```

The generator does not depend on Chrome or Edge. It rejects a document with
fewer than ten pages, a blank text page, or extractable text outside a page
boundary. This utility is only for producing the required PDF; it is not part
of model training, inference, or locked result construction.

## Reproduce

Pass the unmodified official archive and a new output directory:

```bash
python code/main.py verify --data /data1/songwentao/DL/data/data_A.zip --output /data1/contest1-verify
```

The direct shell launcher remains equivalent and is retained for transparent
review:

```bash
bash run_verify.sh /data1/songwentao/DL/data/data_A.zip /data1/contest1-verify
```

For the direct build without the prior static audit:

```bash
bash run_inference.sh /data1/songwentao/DL/data/data_A.zip /data1/contest1-output
```

The output directory must not already exist. Successful execution writes
`result.zip` and `build_audit.json`. The result ZIP is required to have
SHA-256:

```text
d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce
```

The builder rejects a changed official archive, changed base artifact,
changed checkpoint, changed locked rule, noncanonical intermediate CSV, and
any final output that does not match the fixed SHA-256. It writes only under
the caller-provided output path.

Before reconstruction, `run_inference.sh` verifies Python 3.10, Jittor
1.3.10.0, NumPy 1.26.4, Pandas 2.2.3, Numba 0.66.0,
`nvidia-cudnn-cu12` 8.9.7.29, CUDA compiler availability, Jittor CUDA, and a
Jittor arithmetic probe. `run_verify.sh` additionally validates the official
`data_A.zip` SHA-256 before invoking that reconstruction.

## Raw training and fresh inference

The complete Jittor training protocol is available for review and a fresh
official-data run. Use new, disjoint output directories:

```bash
python code/main.py train --data /path/to/data_A.zip --models /path/to/model_output --dataset all --cuda
python code/main.py fresh-infer --data /path/to/data_A.zip --models /path/to/model_output --output /path/to/fresh_output
# Or run the two stages, plus source/output receipts, as one command:
python code/main.py raw --data /path/to/data_A.zip --models /path/to/model_output --output /path/to/fresh_output
```

Dataset1 trains seeds `20260705` and `20260715` with `groups=80000`,
`valid=20000`, `epochs=16`, and `batch=512`. Dataset2 includes MultVAE,
RecVAE, BM25-BPR, pool/set/Transformer and warm-residual Jittor components;
the production BPR uses 256 factors, 20 epochs, batch 32768, 8 negatives and
learning rate 0.002. The community BPR uses 32 factors, 3 epochs, batch 32768,
4 negatives and learning rate 0.002.

The raw protocol records its model and output manifests. The A-list equality
claim applies to `run_verify.sh`, which checks the fixed reconstruction inputs
and final result hash. Historical experiments were executed through automated
job scheduling and parameter search; a small number of edge-candidate
parameter states were not individually preserved. The retained base artifact,
BPR32 checkpoint, locked rules and hash contract are sufficient for the
recorded-result reconstruction and are not a claim that a newly trained
checkpoint has identical bytes to the historical snapshot.

## Frozen algorithm

Dataset1 adds the locked source-support adjustment to the frozen base score.
Dataset2 computes:

```math
z=
\mathrm{qnorm}(\log p_{\mathrm{base}})
+0.05\,\mathrm{qnorm}(I_{\mathrm{exact}})
+0.02\,\mathrm{qnorm}(I_{\mathrm{community}}).
```

The final community term groups candidate occurrences by time and candidate
identifier, excludes same-source and unknown-user pairs, uses Jittor to
normalize retained 32-dimensional BPR user vectors and calculate cosine
similarities, then marks the upper 10 percent. All constants and input hashes
are pinned in the builder. The official test candidate structure is used as
an unlabeled structure only; all output acceptance is by the fixed result
hash, not by a local validation or online-score proxy.

## Known boundary

The package is under 100 MiB because it omits the final output ZIP; that ZIP
is regenerated by `run_inference.sh`.
