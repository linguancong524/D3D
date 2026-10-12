# Benchmarks

D3D evaluates the following public benchmarks. Dataset files are downloaded
and cached at runtime; they are not stored in this repository.

| ID | Dataset | Split | Source |
|---|---|---|---|
| `gsm8k` | GSM8K | `test` | `openai/gsm8k` (`main`) |
| `math500` | MATH-500 | `test` | `HuggingFaceH4/MATH-500` |
| `humaneval` | HumanEval | `test` | `openai/openai_humaneval` |
| `mbpp` | MBPP | `test` | `google-research-datasets/mbpp` (`sanitized`) |
| `mt-bench` | MT-Bench prompts | `train` | `HuggingFaceH4/mt_bench_prompts` |
| `aime25` | AIME 2025 | `train` | `MathArena/aime_2025` |
| `livecodebench-v6` | LiveCodeBench v6 | `test6.jsonl` | `livecodebench/code_generation_lite` |

## Speed protocol

- 2 × NVIDIA H800
- Target on `cuda:0`; draft on `cuda:1`
- `bfloat16`, temperature `0`, `max_new_tokens=2048`
- First 32 samples per dataset, or all available rows if the split is shorter
- Warmup, model loading, and tokenization excluded from timed decoding
- Concurrent request counts: `1`, `4`, `8`, and `16`

Each result should record the package versions, CUDA/PyTorch build, GPU names,
model revisions, dataset revisions, seed, configuration, and source commit.

Dataset files are cached under `D3D_DATA_CACHE` when local JSONL caches are
provided. Otherwise the standard Hugging Face cache is used; the repository
does not redistribute benchmark data.

## Baseline and headline metrics

Quick starts and benchmarks first measure the target model alone:

- The baseline uses the same target checkpoint, prompts, token budget, greedy
  decoding settings, and concurrency as the D3D run.
- It decodes autoregressively on the verifier GPU only (`cuda:0` by default);
  the draft GPU stays idle, so the baseline is single-card AR throughput.
- `speedup` compares the same boundary on both sides: decode-only for
  single-request runs (prefill and the prefill-emitted first target token
  excluded from tokens and time) and end-to-end for concurrent runs (prefill,
  admission, decode and drain included).

Summaries then carry the two headline numbers:

| Field | Meaning |
|---|---|
| `baseline_tps` | Target-only AR tokens per second |
| `speedup` | D3D tokens per second divided by `baseline_tps` |
| `accept_length` | Mean tokens committed per verify step |

Single-request summaries group them in a `headline` block; concurrent benchmark
cells add `baseline_tps`, `speedup`, and `accept_length` fields and also
collect them under the payload `summary`. Set `D3D_BASELINE=0` (scripts) or
`--no-baseline` (Python entry points) to skip the baseline; `speedup` is then
left empty and the D3D run itself is unchanged.

## Metric boundaries

Single-request `aggregate_decode_tokens_per_s` excludes target/draft prefill
and the first target token emitted during prefill. Its clock starts immediately
before the first draft recovery; `output_tokens` still counts every emitted token.
Concurrent `tps` includes prefill and request admission; do not compare the two
numbers as if they had identical timing boundaries. Both exclude model loading,
warmup, and tokenization. Token counts use actual emitted output tokens and EOS
is respected by default.

MT-Bench uses the first turn for throughput measurement, not the full multi-turn
judge evaluation. HumanEval, MBPP, and LiveCodeBench outputs are generated but
not executed or graded by these speed scripts. Speed runs are not quality
benchmarks or claims of pass@1. GSM8K/MATH answer matching is diagnostic, not a
replacement for the paper's complete quality evaluation protocol.

Override datasets with `D3D_DATASETS="math500 humaneval"`; select a local or
Hub checkpoint with `D3D_TARGET_MODEL` and `D3D_DRAFT_MODEL`. The scripts use
block size 16, one denoising step, three backup anchors, greedy decoding, and
strictly causal draft-prefix attention (matching the serving backend), with
thinking disabled. `D3D_MAX_SAMPLES` changes the sample cap.
`D3D_MAX_NEW_TOKENS` overrides the default 2048-token limit for smoke runs;
these shorter runs are not formal protocol results. The concurrent
script accepts `D3D_CONCURRENCY="1 4 8 16"`.

For larger target models, the paged KV cache may need a lower `D3D_PAGES`
value to leave room for weights and temporary buffers. The default is 512
pages per device; a 32B target can require a smaller pool even on an 80 GB
GPU. The runner checks that the requested context and worst-case branch page
reservations fit the configured capacity before model loading. Report the
chosen value and measure peak memory before formal runs.
