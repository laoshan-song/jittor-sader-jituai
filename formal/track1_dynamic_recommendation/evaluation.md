# Track 1 Evaluation System

The evaluation system has three layers:

1. Submission audit: checks generated zips on official test candidates.
2. Offline proxy MRR: creates labeled train-only validation tasks.
3. Online calibration: compares audit metrics with known leaderboard scores.

The online leaderboard is the only trusted score. Local metrics are used to
filter bad candidates and decide which experiments are worth a daily submission.

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

## GPU Policy for Autoresearch

Use one GPU per training process:

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

Do not spread one light experiment over many GPUs. Start parallel experiments
only when one experiment is already using a full card efficiently, or when
running separate one-GPU jobs that each keep their assigned card busy.
