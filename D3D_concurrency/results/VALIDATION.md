# Validation of the initial implementation

- Four independent Math500 prompts, 32 tokens each: the multi-request target
  outputs match Hugging Face Qwen3-8B greedy generation token for token.
- Original D3DV2 SDAR comparison on one prompt: recovery block matches;
  all 46 candidate keys and all 736 packed candidate tokens match exactly.
- Forty requests complete at each of concurrency 8, 16 and 32, exercising
  admission, refill, individual acceptance lengths and independent KV pages.
- Four CPU unit tests pass: variable-prefix packed visibility, recovery/padding
  isolation, page rollback/release, and greedy argmax versus top-k tie handling.
- All 12 benchmark cells pass the CPU integrity audit: actual output counts,
  wall-time TPS, request identities, concurrency caps, EOS trimming and identical
  input-workload hashes across concurrency levels.

## Floating-point scope

Bitwise invariance across batch sizes is not guaranteed. Of the 40 short smoke
outputs, 39/40 at C=16 and 38/40 at C=32 match C=8 exactly. The first differing
positions are documented in `batch_numerics.json`. Independent HF teacher-forced
logits at the same prefixes show:

| Prompt index | Output position (zero-based) | Competing logits |
|---:|---:|---|
| 32 | 10 | 34.0 vs 34.0 (exact BF16 tie) |
| 38 | 31 | 41.5 vs 41.75 (one BF16 representable step) |

These observations are consistent with batch/kernel floating-point differences;
they do not establish full-dataset accuracy or bitwise equivalence. Throughput
uses each run's actual emitted token count. No quality or AR speedup baseline
is reported.

## Evidence

- `smoke_concurrency.json`: successful GPU reference and concurrency checks.
- `unit.log`: four passing unit tests.
- `batch_numerics.json`: numerical diagnosis, run after throughput measurements.
- `table_1_7b_8b.json`: raw outputs and timing for all 12 measured cells.
- `dependency_sources.json`: hashes of reused local D3DV2/SDAR sources.

Recheck the table artifact with:

```bash
python audit_results.py results/table_1_7b_8b.json
```
