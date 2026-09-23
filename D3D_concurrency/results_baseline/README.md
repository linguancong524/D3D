# AR concurrency baselines (no draft), n=32 subsets

Two verifier sizes, same harness (`scripts/baseline_concurrency.sh` ->
`baseline_concurrency.py`), same protocol:

* `8B/` — Qwen3-8B. Measured 2026-09-21, run `20260921_081947_n32_t512_r2`
  (24 measurements, audit passed) plus the MT-Bench C=1 confirmation
  `20260921_093838_mtbench_c1_confirm`. See `8B/README.md` for the full
  protocol description and `8B/summary_n32_t512.md` for the table.
* `32B/` — Qwen3-32B. Empty until measured; command below.

Protocol in one line: greedy BF16, thinking disabled, natural EOS, 512-token
cap, first 32 prompts per benchmark (MT-Bench first turn only), 256-token
full-workload warmup discarded, two attempts per cell with one automatic
retest when the slower attempt is below 85% of the faster one, timing includes
prefill/decode/drain but excludes loading, tokenization, warmup and
serialization. Baselines stay on the n=32 subset because their speed is stable;
the speculative suite in `../results_full` uses full samples.

## Qwen3-32B command

Run it in `D3D_concurrency` on an **80 GB card**. `--pages 96` is required:
Qwen3-32B is ~62 GB of bf16 weights and the default 512-page KV pool would add
another 34 GB.

`baseline_concurrency.py` loads a second full copy of the verifier as an
independent HF greedy reference, so on a single card the two copies cannot
coexist (62 + 62 GB). Either point the reference at another visible card, or
skip it. Skipped runs record `skip_reference_validation` in `benchmark.json`
and an empty `validation.json`.

```bash
cd /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3D_concurrency
export CUDA_VISIBLE_DEVICES=0,1          # 0 = measured verifier, 1 = HF reference
RUN_ID=$(date -u +%Y%m%d_%H%M%S)_32b_n32_t512_r2
for ds in math500 humaneval mt-bench; do
  mkdir -p "results_baseline/32B/$ds"
  bash scripts/baseline_concurrency.sh \
    --target /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-32B \
    --datasets "$ds" --concurrency 1 4 8 16 --max-samples 32 \
    --max-new-tokens 512 --warmup-new-tokens 256 --repeats 2 \
    --max-retests 1 --slow-fraction 0.85 --pages 96 \
    --reference-device cuda:1 \
    --output-root results_baseline/32B --run-id "$RUN_ID" \
    >"results_baseline/32B/$ds/${RUN_ID}_driver.log" 2>&1
done
```

Single-card variant: drop `CUDA_VISIBLE_DEVICES`, use `CUDA_VISIBLE_DEVICES=<card>`
and replace `--reference-device cuda:1` with `--skip-reference-validation`.

Expected cost: ~4 hours (C=1 dominates — the 32B verifier runs at roughly
10 output tokens/s there, so one math500 C=1 attempt is ~22 min plus a 13 min
warmup). The three datasets load the 32B model three times (~4 min each).
