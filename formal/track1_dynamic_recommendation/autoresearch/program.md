# Track 1 Autoresearch Program

This directory adapts the `karpathy/autoresearch` idea to the Track 1 dynamic
recommendation leaderboard. The goal is not to optimize a misleading local MRR
as hard as possible. The goal is to run small, reproducible experiments, reject
obviously unsafe submissions, and keep a clear trail of what changed.

## Loop

Each experiment should follow the same loop:

1. Pick one narrow hypothesis.
2. Run one command that produces a submission zip.
3. Check the zip format and rank-quality guardrails.
4. Record the command, parameters, artifacts, and metrics.
5. Promote only one candidate for online submission.

## Guardrails

The A leaderboard is the only trusted score. Local validation is a proxy and is
known to be distribution-shifted, so the runner reports guardrails instead of a
single "best local score".

Required checks:

- `dataset1.csv` and `dataset2.csv` are present.
- Row counts match the official test files.
- Every row has exactly 100 probabilities in `[0, 1]`.
- Rounded probabilities preserve useful ordering signal. Avoid submissions
  where most rows collapse to one high value and 99 identical low values.
- Ranking should not drift too far from `result_sequence.zip` unless there is a
  clear reason and an online score confirms the direction.

## Safe Search Space

Start from conservative model blends:

- `mf_weight`: `0.5`, `1`, `1.5`, `2`, `3`
- `hard_negative_ratio`: `0`
- `probability_mode`: `rank`
- `epochs`: `6`

Hard-negative experiments are allowed only after a safer proxy validation is
added. They previously produced excellent local validation metrics but worse
online behavior.

## Promotion Rule

Prefer the candidate with:

1. Valid format.
2. 100 distinct probabilities per row on average.
3. Higher agreement with the stable sequence baseline when no online evidence
   says otherwise.
4. Smaller model weight when two candidates look similar.

Daily submissions are limited, so the runner should create candidates, but a
human should choose the actual upload after inspecting the summary.
