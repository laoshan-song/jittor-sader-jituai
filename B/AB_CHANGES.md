# A/B algorithm consistency

The accepted A-list and B-list packages share the same outer contract:
official competition input, retained final state, Jittor candidate scoring,
candidate-local fusion, and deterministic ZIP serialization.

| Contract | A list | B list |
| --- | --- | --- |
| Base state | retained score archive | retained q35/q7 frozen checkpoint |
| Full reconstruction | official-data raw training | official-data C2/C3/C5/C6/RUC4/third_1 graph |
| Final learned member | BPR32 signal | MF32 signal |
| Candidate processing | bounded local residual | bounded local residual and rank grid |
| Output | deterministic two-member ZIP | deterministic two-member ZIP |

The B-list package exposes two commands only:

```text
verify     frozen final-layer reproduction
reproduce  complete data_B.zip reconstruction
```

The full B-list graph is implemented in `code/pipeline/`. It trains the D3/D4
components, writes base score matrices, serializes q35/q7 frozen scores, and
constructs the recorded result. `README.md` and `code/pipeline/README.md`
describe the full stage graph and output contract.
