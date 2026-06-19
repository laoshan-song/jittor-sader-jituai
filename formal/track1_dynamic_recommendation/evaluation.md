# Track 1 Evaluation System

The evaluation system has three layers:

1. Submission audit: checks generated zips on official test candidates.
2. Offline proxy MRR: creates labeled train-only validation tasks.
3. Online calibration: compares audit metrics with known leaderboard scores.

The online leaderboard is the only trusted score. Local metrics are used to
filter bad candidates and decide which experiments are worth a daily submission.
After the failed `result_calibrated_d2_medium.zip` submission scored
`0.9764773175367665`, local train-split MRR is no longer allowed to approve a
submission by itself. It may only be used for diagnosis. The pre-submit gate is
the online-calibrated ranking drift check against the best known online anchor.

## Submission Audit

```bash
python formal/track1_dynamic_recommendation/evaluate_submission.py \
  --data-zip ../data_A.zip \
  --baseline-zip outputs/track1/result.zip \
  --submissions \
    outputs/track1/result.zip \
    outputs/track1/result_mf_final.zip \
    outputs/track1/result_mf_hardneg_final.zip \
    outputs/track1/result_mf_conservative_rank_w1.zip \
  --output-json outputs/track1/online_calibration_audit.json
```

Important fields:

- `bad_rows`: must be `0`.
- `avg_distinct`: higher is better. Very low values mean the rounded submission
  destroyed most ranking information.
- `avg_max_probability`: lower is safer for MRR-style ranking when probabilities
  are not directly scored.
- `avg_normalized_entropy`: very low entropy is a warning sign.
- `top10_jaccard`: useful as a drift guardrail, not as the main optimization
  target.

## Pre-Submit Gate

Use the current best online-verified package as the anchor:

```bash
python formal/track1_dynamic_recommendation/submission_gate.py \
  --anchor outputs/track1/result.zip \
  --candidate /path/to/candidate.zip
```

Default hard gates:

- `dataset1 top1 >= 0.99`
- `dataset1 top10_jaccard >= 0.95`
- `dataset2 top1 >= 0.90`
- `dataset2 top10_jaccard >= 0.75`

These thresholds are calibrated from known online outcomes:

- `result_mf_conservative_rank_w1.zip`: online `1.1099002959299407`,
  `dataset2 top1=0.9691`, `dataset2 top10_jaccard=0.8211` versus the current
  anchor.
- `result_calibrated_d2_medium.zip`: online `0.9764773175367665`,
  `dataset2 top1=0.5313`, `dataset2 top10_jaccard=0.3344`; it must be blocked.

## Online Calibration

Create a CSV like `online_scores.example.csv` with leaderboard scores:

```csv
submission,online_score,notes
result.zip,1.0006084897236325,initial baseline submission
```

Then run:

```bash
python formal/track1_dynamic_recommendation/calibrate_online.py \
  --audit-json outputs/track1/online_calibration_audit.json \
  --online-scores formal/track1_dynamic_recommendation/online_scores.example.csv \
  --output-json outputs/track1/online_calibration.json \
  --output-md outputs/track1/online_calibration.md
```

Current calibration says the strongest local indicators are:

- high `avg_distinct`
- high `avg_entropy`
- low `avg_max_probability`

Agreement with `result.zip` is only a guardrail because `result.zip` is not the
best known online submission.

## Offline Proxy MRR

```bash
python formal/track1_dynamic_recommendation/offline_eval.py \
  --data-zip ../data_A.zip \
  --sample-positives 1000 \
  --splits temporal,official \
  --candidate-strategies mixed,hard \
  --output-json outputs/track1/offline_eval_core_sample.json
```

Use this to understand model behavior on labeled train-only splits. Do not pick
submissions from offline MRR alone.

The `official` split in `dataset2/train.csv` does not match the hidden test
candidate distribution closely enough. It can rank an experiment higher locally
while the leaderboard falls sharply, so any candidate that fails the
pre-submit gate is rejected even if offline MRR improves.

## GPU Policy for Autoresearch

Use one GPU per training process:

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

Do not spread one light experiment over many GPUs. Start parallel experiments
only when one experiment is already using a full card efficiently, or when
running separate one-GPU jobs that each keep their assigned card busy.
