# A/B algorithm consistency

A-list and B-list are the same algorithm on Track 1's temporal-graph
candidate-ranking task. Both build history-only members, calibrate every model
inside the 100-candidate row with `qnorm`, fuse with bounded candidate-local
residuals over a retained frozen base, and serialize a deterministic two-member
`result.zip` accepted by a fixed SHA-256. The method, the candidate boundary,
the fusion principle, and the audit contract are identical.

| Aspect | A list | B list | Shared design |
| --- | --- | --- | --- |
| Task | source + time + 100 candidates | source + time + 100 candidates | rank inside the given candidates |
| Data boundary | history-only, no future edges | history-only, no future edges | no test labels, no external data |
| Members | Dataset1/Dataset2 graph rank + VAE/BPR + set experts | Dataset3 frequency residual + Dataset4 expert graph and MF | learn or count from official history |
| Base + fusion | frozen base + bounded in-row residual | frozen base + bounded in-row residual | correct only within the 100 candidates |
| Output | deterministic two-member ZIP | deterministic two-member ZIP | fixed order, fixed digits, SHA-256 |
| Delivery | verify from retained final state | verify + full-chain fresh reproduce | fixed public interfaces and audited output |

## Differences are data-scale adaptations only

The B-list dataset is larger (more sources/items, larger query set), so the same
algorithm is instantiated with engineering adaptations that do not change the
model design: expanded entity vocabularies with sorted-vocabulary index mapping,
chunked reading and streaming batches for the larger candidate volume, and more
member seeds/stages wired through the same in-row fusion. Dataset3 stays a
target-frequency structural residual and Dataset4 stays an implicit-MF member;
both feed the same in-row calibration and deterministic serialization used on
the A list. These are scale adaptations, not a new architecture.
