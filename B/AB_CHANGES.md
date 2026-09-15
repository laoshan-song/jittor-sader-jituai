# A/B algorithm consistency

A-list and B-list share the same Track 1 candidate-ranking framework. Both
build history-only members, calibrate models inside the 100-candidate row with
`qnorm`, fuse bounded candidate-local residuals, and serialize a deterministic
two-member `result.zip`. Their data boundary, candidate-local fusion principle,
Jittor implementation, and auditable output contract are shared; the concrete
model members and reproduction mechanisms are dataset-specific.

| Aspect | A list | B list | Shared design |
| --- | --- | --- | --- |
| Task | source + time + 100 candidates | source + time + 100 candidates | rank inside the given candidates |
| Data boundary | history-only, no future edges | history-only, no future edges | no test labels, no external data |
| Members | Dataset1/Dataset2 graph rank + VAE/BPR + set experts | Dataset3 nine-member/C2-C6/RUC4 graph + Dataset4 temporal/MF/session/meta graph | learn or count from official history |
| Base + fusion | retained base + bounded in-row residual | staged fresh base + bounded in-row residual | correct only within the 100 candidates |
| Output | deterministic two-member ZIP | deterministic two-member ZIP | fixed order, fixed digits, SHA-256 |
| Delivery | verify from retained final state | verify + full-chain fresh reproduce | fixed public interfaces and audited output |

## Dataset-specific configuration

The B-list dataset is larger (more sources/items, larger query set), so the
shared framework is instantiated with expanded entity vocabularies,
sorted-vocabulary index mapping, chunked reading and streaming batches, and
more member seeds/stages wired through the same in-row fusion. Dataset3
expands the same graph-ranking and set-model principles through nine base
members and C2/C3/C5/C6/RUC4 stages; Dataset4 expands temporal, MF,
session-graph, and meta members. Both feed the same candidate-local
calibration and deterministic serialization used on the A list. These
differences configure the shared framework for B-list fields and scale; they
do not change the task or candidate boundary.
