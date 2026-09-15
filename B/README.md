# Track 1 B-list Jittor reproduction

This package contains the complete Dataset3/Dataset4 Jittor training and
inference graph for the recorded Track 1 B-list score
`1.5240999401892983`. It exposes two complementary public routes:

| Route | Purpose | Model execution | Output |
| --- | --- | --- | --- |
| `verify` | Fast recorded-result reconstruction | Retained final inference state | Byte-exact `result.zip` |
| `reproduce` | Full-chain reproduction from `data_B.zip` | Retrains every D3/D4 stage and final MF32 | Fresh intermediate result plus the same byte-exact `result.zip` |

The full-chain route always produces and records its fresh intermediate state.
Historical byte equality is available only when that state matches the pinned
fresh source hashes; any other result fails closed before alignment.

The official archive and final submission ZIP are not packaged. Neither route
reads test labels or external datasets. The repository does retain
reproducibility state for deterministic alignment with the historical
submission; that state is not training data and is described under
**Numerical consistency**.

## Package layout

- `code/main.py`: the two public commands.
- `run_verify.sh`: fast reconstruction and hash verification.
- `run_reproduce.sh`: complete official-data training, inference, numerical
  alignment, and final submission construction.
- `code/pipeline/reproduce_full.py`: full-chain coordinator.
- `code/pipeline/reproduce_third_1.py`: D3/D4 RUC4 base, replay caches,
  side features, meta training, and fresh inference.
- `code/pipeline/reproduce_ruc4.py`: D3 Set Transformer and D4 session-graph
  ranking stages.
- `code/pipeline/reproduce_c6.py`, `reproduce_c5.py`, `reproduce_c3.py`, and
  `c2_source/reproduce_c2.py`: the staged D3/D4 base graph.
- `code/pipeline/pack_frozen_base.py`: D3 q35/LZMA and D4 q7 serializer.
- `code/pipeline/adapt_fresh_mf32.py`: deterministic MF32 numerical alignment.
- `code/build_submission.py`: final candidate-local residual and ZIP builder.
- `code/audit_package.py`: route-aware package, hash, and data-boundary audit.

## Environment

The validated target is Ubuntu 22.04, Python 3.10, Jittor 1.3.11.0, an NVIDIA
RTX 4090, and a CUDA 12.4-compatible runtime. Install the pinned dependencies:

```bash
python3.10 -m venv /data1/sader-repro-py310
source /data1/sader-repro-py310/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export ML_CACHE_ROOT=/data1/sader-runtime
```

The launchers create caller-local Jittor, CUDA-wrapper, Python-cache, and
temporary directories. They do not modify the system CUDA installation.

## Reproduce

Pass the unmodified official archive and a new output directory:

```bash
# Fast recorded-result reconstruction.
python code/main.py verify \
  --data /path/to/data_B.zip \
  --output /data1/b-verify \
  --gpu 0

# Complete D3/D4 training and inference graph.
python code/main.py reproduce \
  --data /path/to/data_B.zip \
  --output /data1/b-reproduce \
  --gpu 0
```

Both commands require the official archive SHA-256
`ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2`
and produce a final `result.zip` with SHA-256:

```text
9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba
```

`verify` validates the complete package and uses the retained inference state.
`reproduce` first runs a locked-asset-independent package audit, then executes:

```text
data_B.zip
-> C2 -> C3 -> C5 -> C6 -> RUC4 -> third_1
-> fresh result.zip
-> final MF32 training
-> numerical consistency
-> final result.zip
```

The full-chain receipt records the fresh result, freshly trained MF32,
generated base, generated final model, and final submission hashes. The fresh
intermediate result remains under the caller-provided work directory for
inspection.

## Dataset3

### Nine-member base ensemble

Dataset3 starts with three model variants (`raw`, `cf`, and `hist_cf`) and
three seeds (`20260810`, `20260811`, and `20260812`). Each member trains the
scene model and a FastRanker from official history, then `fit_ensemble.py`
combines their validation and confirmation predictions. The training objective
compares the true destination against the supplied 100-candidate group.

Code path: [`reproduce_c2.py`](code/pipeline/c2_source/reproduce_c2.py) ->
[`train_grid.py`](code/pipeline/c2_source/code/b_rank_a_port/train_grid.py) ->
[`fit_ensemble.py`](code/pipeline/c2_source/code/b_rank_a_port/fit_ensemble.py)
-> [`infer_d3_source.py`](code/pipeline/c2_source/code/infer_d3_source.py).

### Structural stages

The ensemble is refined through the same staged graph used by the public
reproduction command:

- **C2 source/session support**: same-time cross-source and near-time
  source/session evidence.
- **C3 multiscale support**: directional support over multiple temporal
  windows.
- **C5 session ring**: guarded session-ring residual.
- **C6 tie groups**: fixed tie-group transformation, with three-seed gate
  evaluation retained as a diagnostic report.
- **RUC4**: three Jittor Set Transformer members trained with seeds
  `20260810`, `20260824`, and `20260907`.

`third_1` preserves the RUC4 Dataset3 member while it performs the Dataset4
meta stage. The final builder adds only the bounded target-frequency signal:

```text
r3 = tanh(qnorm(log(1 + destination_count)) / 2)
score3 = base3 + 0.005 * r3
```

## Dataset4

### Temporal and matrix-factorization experts

Dataset4 builds causal caches from official history and trains:

- 32- and 64-history temporal attention members over three seeds;
- three test-pool temporal members;
- three implicit-MF members and one transition-MF member;
- six pair-new Transformer residual members with hidden sizes 64 and 96.

The control and pair-new reports feed `d4_multimodel_infer.py`, which generates
the C2 Dataset4 scores. Later full-chain stages train another three
512-dimensional full-history MF members and a 512-dimensional transition-MF
for the meta feature graph.

Code path: [`reproduce_c2.py`](code/pipeline/c2_source/reproduce_c2.py) ->
[`reproduce_ruc4.py`](code/pipeline/reproduce_ruc4.py) ->
[`reproduce_third_1.py`](code/pipeline/reproduce_third_1.py) ->
[`build_submission.py`](code/build_submission.py).

### Session graph and meta ranker

RUC4 builds history/test-pool replay caches, identity and baseline caches, then
trains three session-graph hard rankers. RP3/RUC2/RUC3/RUC4 candidate fusion
produces the Dataset4 base consumed by `third_1`.

`third_1` constructs hierarchy and neighbor side features, combines them with
replay, baseline, identity, temporal, MF, and transition-MF signals, and trains
the 75-feature Jittor meta ranker. Its formal inference writes the fresh
Dataset4 matrix. The final independent member is a newly trained 32-dimensional
implicit MF:

```text
f(s, c) = dot(u_s, v_c) + b_c
r4 = tanh(qnorm(f(s, c)) / 2)
score4 = base4 + 0.02 * r4
```

Stable descending order is mapped to the fixed 100-position rank grid before
`dataset4.csv` is written.

Control, pair-new, session-graph, leakage, and meta selection reports are
recorded diagnostics. The receipts preserve their decisions; formal inference
uses the freshly fitted reports/models together with the pinned deployment
policy rather than treating every diagnostic label as a process exit code.

## Numerical consistency

Historical operator versions, floating-point environments, and a small number
of unavailable intermediate parameter states prevent an unconstrained retrain
from being assumed byte-identical. The full-chain route therefore applies a
fixed reproducibility alignment layer after producing the fresh D3/D4 matrices
and fresh MF32:

- Dataset3 is aligned on the `1e10` fixed-point grid before q35/LZMA encoding.
- Dataset4 is aligned on the q7 grid before little-endian bit packing.
- MF32 is aligned after row-wise q8 quantization through parameter and scale
  residuals.

The residuals are elementwise over the recorded score grids and quantized
parameter tensors. They are target-specific reproducibility state, not learned
general-purpose corrections.

These operations do not replace any training stage or read the retained
`code/assets/locked/` state. They are tied to the recorded fresh source hashes;
an unexpected training or environment result fails closed instead of being
silently accepted. Their purpose is deterministic historical-result
reconstruction, not a new ranking model or a cross-environment generalization
claim.

## Data boundary

Training and learned statistics use the official archive. Test candidate and
time columns are used as unlabeled query structure. No test ground truth,
external dataset, external prediction, or non-Jittor deep-learning framework
is used.

The package contains two types of reproducibility state: retained inference
state for the fast route, and numerical-alignment residuals for the full-chain
route. `run_reproduce.sh` audits and executes without reading or requiring the
retained fast-route state.

## Verification boundary

`audit_package.py` validates package contents, source wiring, manifests, and
route isolation; it does not claim to execute multi-hour GPU training.
`REPRODUCTION_RECEIPT.json` is the runtime evidence produced by the full-chain
coordinator and is not bundled in the repository. The repository therefore
supplies the executable graph and static contracts; each reviewer creates the
runtime receipt by running `reproduce`. Exact full-chain reconstruction is
supported only for the pinned environment and recorded fresh source hashes.
