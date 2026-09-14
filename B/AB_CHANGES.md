# A/B algorithm consistency

The accepted A-list package and this B-list package use the same high-level
reproduction contract: official competition input, retained model assets,
Jittor candidate scoring, candidate-local fusion, and deterministic result
serialization.

| Contract | A list | B list adaptation |
| --- | --- | --- |
| Fixed base member | retained `base_result.zip` | compact retained base checkpoint |
| Base origin | official-data raw training | official-data full pipeline in `code/pipeline/` |
| Jittor model | BPR32 candidate signal | MF32 source-candidate signal |
| Candidate processing | bounded candidate-local residual | bounded candidate-local residual |
| Data organization | A-list entity tables and batches | expanded entity tables and streaming batches |
| Output | deterministic two-member ZIP | deterministic two-member ZIP |
| Training interface | official-data raw training | official-data MF32 training and base generation |

The B-list implementation expands the entity vocabulary and uses streaming
batches for the larger query set. Dataset3 uses an official-training-data
target-frequency statistic; Dataset4 uses a 32-dimensional Jittor implicit
preference model. Both feed the same candidate-local fusion and deterministic
serialization stages.

## Frozen base generation

The retained frozen base is no longer opaque: `code/pipeline/` reconstructs it
from the official data. `code/pipeline/reproduce_third_1.py` trains every
Dataset3/Dataset4 component from scratch and emits the base score matrices, and
`code/pipeline/pack_frozen_base.py` — the exact inverse of the
`code/build_submission.py` decoders — packs them into `frozen_base.ckpt`
(Dataset3 zig-zag q35+LZMA, Dataset4 7-bit packing). This makes the base an
end-to-end product of official data rather than a fixed input.

This makes the frozen base a product of the official-data generation chain
rather than a fixed input, so the recorded top submission is reproducible from
the data alone.

For the recorded B-list result, the reviewer supplies the official
`data_B.zip` and runs `run_verify.sh`. The fixed base and Jittor checkpoint
are package members validated by SHA-256 and consumed by the result builder.

The supplementary commands preserve the same source layout and interfaces:

```bash
bash run_train.sh /path/to/data_B.zip /path/to/model-output 0
bash run_fresh_inference.sh /path/to/data_B.zip \
  /path/to/model-output /path/to/result-output 0
```

`A_LIST_REFERENCE.md` records the accepted A-list archive and relevant member
hashes so the shared official-input, retained-asset, Jittor-inference, and
deterministic-output contract is directly traceable.
