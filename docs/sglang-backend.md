# Two-GPU SGLang backend

## Default: upstream Hybrid

The default backend uses the original `DECOUPLED` Hybrid architecture from
`wintem0531/sglang-speedup-nvidia`: GPU 0 holds the Target and a local required
Draft; GPU 1 holds the optional Draft that asynchronously prepares branches.
Uncovered results fall back to the local Draft. This is not strict Target/Draft
separation. The vendored implementation is unchanged; D3D wraps its launcher.

Install the separate environment and start both workers from the D3D root:

```bash
bash scripts/sglang/install.sh
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=lingc8/D3D-1.7B-to-8B
export TARGET_GPU=0 DRAFT_GPU=1 MAX_RUNNING_REQUESTS=1
bash scripts/sglang/launch.sh hybrid
```

From another terminal, using the same model and concurrency variables:

```bash
export TARGET_MODEL=Qwen/Qwen3-8B MAX_RUNNING_REQUESTS=1
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset math500 --tokenizer "$TARGET_MODEL" \
  --samples 128 --concurrency "$MAX_RUNNING_REQUESTS" \
  --repetitions 1 --max-new-tokens 512 \
  --output outputs/sglang/hybrid-c${MAX_RUNNING_REQUESTS}.json
```

Set `MAX_RUNNING_REQUESTS` to `1`, `4`, `8`, or `16` and restart the backend.
Stop Hybrid before starting the native SGLang AR baseline with
`bash scripts/sglang/launch.sh ar`; measure with identical benchmark arguments,
then pass its JSON to the Hybrid benchmark via `--baseline`.
Use a fresh output filename for each run. Datasets download/cache automatically.

Defaults follow the upstream H800 Hybrid graph configuration: block 16,
one anchor, FP16 remote Draft, BF16 Target, local packed graph capacity 1024
tokens, remote packed CUDA Graph enabled, covered-result wait 12 ms,
local prefix reuse disabled. Target verification CUDA Graph is enabled;
prefill CUDA Graph is disabled. Local full replay exceeds graph capacity at
long prefixes or large batches and falls back to eager execution. The Target
request pool reserves one extra padding slot; client `--concurrency` controls
the measured load. `--dry-run` prints commands without starting GPU workers.

`DETERMINISTIC=0` is the default for comparison with ordinary SGLang AR;
set `DETERMINISTIC=1` for diagnostic deterministic Target operators. Neither
mode is an accuracy claim: inspect `exact_text_matches`, and validate answer
quality separately. Non-deterministic multi-request runs can differ from AR.
High-concurrency speedups are not guaranteed. Raw results/logs remain under
ignored `outputs/` and are not public benchmark claims.

The following section documents the retained opt-in strict-remote backend.

## Optional: strict two-GPU separation

The `draft` / `remote` modes preserve D3D's deployment topology: GPU0 owns only the
Target model and its KV cache; GPU1 owns only the Draft model and its KV cache.
There is no Target-local Draft or Hybrid fallback. Native D3D entry points and
their PyTorch 2.6 environment remain unchanged.

## Source and implementation

The SGLang snapshot is from `wintem0531/sglang-speedup-nvidia`, commit
`db686d124840ebfdd61e787b3a7fbf543244b6f0`, based on SGLang `v0.5.18`.
The snapshot retains upstream license files. See
`../third_party/sglang/UPSTREAM_D3D.md` for the export boundary and attribution.
Original benchmark reports in that snapshot are not D3D reproduction results.

The fork's built-in `DECOUPLED` algorithm is Hybrid and always constructs a
Target-local Draft. D3D instead registers a separate `D3D_REMOTE` plugin:

- Target uses SGLang's host-tree verification and ordinary greedy Target
  sampling. The Target process never constructs a Draft worker or Draft KV pool.
- Draft runs the fork's native SDAR `OneStep` engine on the other GPU.
- Required proposals use causal committed-prefix replay plus an
  anchor-conditioned bidirectional diffusion block. All required work is remote.
- After sending a proposal, the Draft service computes packed continuations
  while the Target verifies it. Verification feedback selects the next proposal.
- On an uncovered outcome, the Target waits for remote replay, never falls back
  to a colocated model. Transport failures stop the request rather than silently
  switching topology.

Current scope is single-node, TP1/DP1, concurrency one, greedy decoding. The
endpoint is a trusted localhost-only internal channel, not an authenticated
public serving API. This is an initial integration, not a claim that all model
pairs, sampling modes or high concurrency have been validated.

## Separate environment

Do not install this backend into the native D3D environment. Run from the D3D
root with `uv` available:

```bash
bash scripts/sglang/install.sh
```

`runtime.lock` records the exact initial CUDA-12 runtime assembled for this
reproduction. Installation deliberately uses `--no-deps`: the fork's generic
metadata includes CUDA-13 dependencies and optional architectures that are not
the tested Qwen3/SDAR serving path. The environment is not advertised as a fully
dependency-satisfied installation of every SGLang feature. In particular,
`cuda-python` metadata and installed CUDA-12 bindings differ; prefill CUDA Graph
is disabled. Unused CUDA-13 packages in the captured lock are not a CUDA-13
driver requirement for the tested path. Do not run an unpinned resolver over
this environment: it can replace the CUDA-12 PyTorch build.

The tested packages include Python 3.12, PyTorch `2.13.0+cu129`,
`sglang-kernel==0.4.6.post1+cu129`, Transformers `5.12.1`, FlashInfer `0.6.17`.
Rust extensions are disabled. CUDA JIT kernels still need `nvcc`, a C++ compiler,
and Ninja. Runtime packages are isolated from native D3D; they do not upgrade the
host driver. Check the result report for the actual GPU, driver and toolkit.

## Launch

Set paths or public Hub model IDs. An already downloaded checkpoint can be used
without changing its `config.json` or weights:

```bash
export TARGET_MODEL=Qwen/Qwen3-8B
export DRAFT_MODEL=lingc8/D3D-1.7B-to-8B
export TARGET_GPU=0 DRAFT_GPU=1
export BLOCK_SIZE=16 ANCHORS=1
```

Terminal 1, Draft only on GPU1:

```bash
bash scripts/sglang/launch.sh draft --metrics-file outputs/sglang/draft.json
```

Wait for `READY tcp://127.0.0.1:17891`. Terminal 2, Target only on GPU0:

```bash
bash scripts/sglang/launch.sh remote
```

`D3D_REMOTE` rejects `--speculative-draft-model-path`, preventing accidental
Hybrid placement. The launcher sets an identical Target configuration for
remote and AR: BF16, FA3 attention, greedy inference, no radix
cache, decode CUDA Graph, disabled prefill CUDA Graph, context length 8192 and
max-running-requests one. Draft defaults to BF16 and FlashInfer, with packed
pending replay causal. The launcher enables the Draft's bounded native packed
graph path by default; set `DRAFT_CUDA_GRAPH=0` for an eager ablation. Shapes
outside the captured capacity fall back to eager on the Draft GPU, never on the
Target GPU.

If `MAX_BRANCHES` is unset, the launcher uses `BLOCK_SIZE * ANCHORS` rather than
silently truncating anchor counts above three to 48 branches. Draft graph token
capacity defaults to at least 1024, rounded up to a power of two sufficient for
the branch blocks and the pending proposal. Override it with
`--packed-graph-max-tokens` when needed. A long uncached prefix can still exceed
that capacity and use remote eager replay.

The remote host-proposal worker uses a non-overlap Target scheduler so that
committed token IDs are available before each RPC, as required by the fork's
host-proposal path. This does not disable cross-GPU asynchronous drafting or
Target CUDA Graph. Native AR retains SGLang's normal overlap scheduler; it is
not slowed down to match a speculative implementation limitation.

`DETERMINISTIC=1` is an optional diagnostic configuration, not the performance
default. SGLang's deterministic flag changes GEMM and other operators to
batch-invariant implementations in addition to selecting PyTorch sampling.
The initial deterministic AR run was approximately 44 token/s even with decode
CUDA Graph enabled; the standard run without that flag was approximately
152 token/s on the same hardware and workload. Never mix these configurations
in a speedup denominator. Greedy temperature zero does not itself require the
deterministic-inference flag; output agreement is measured separately.

## Matched AR comparison

Stop the remote Target before launching AR on the same GPU:

```bash
bash scripts/sglang/launch.sh ar
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset /path/to/math500-test.jsonl --tokenizer "$TARGET_MODEL" \
  --samples 8 --repetitions 3 --max-new-tokens 512 \
  --output outputs/sglang/ar.json
```

Then stop AR, launch `remote`, and run exactly the same workload:

```bash
.venv-sglang/bin/python scripts/sglang/benchmark.py \
  --dataset /path/to/math500-test.jsonl --tokenizer "$TARGET_MODEL" \
  --samples 8 --repetitions 3 --max-new-tokens 512 \
  --baseline outputs/sglang/ar.json --output outputs/sglang/remote.json
```

Both sides use identical sampled source indices, rendered chat token IDs,
thinking disabled, temperature zero, natural EOS, one disjoint warmup and the
same token cap. Existing output files are not overwritten. Prompt rendering and
model startup are outside timing; HTTP/RPC, Target prefill, drafting waits,
verification and detokenization are inside E2E timing.

- `speedup`: aggregate remote output tokens/second divided by matched native
  SGLang AR output tokens/second. AR uses one GPU; D3D uses two, not equal GPU
  resource budgets.
- `acclen`: total generated tokens minus one prefill token per request, divided
  by total Target verification calls. Includes the Target bonus token; subtract
  one for accepted Draft-only tokens per call. EOS/token-cap trimming can reduce
  the last verification's contribution.
- `exact_text_matches`: checks greedy output equivalence against AR, not math
  accuracy. This small performance sample is not a full benchmark or accuracy
  evaluation.

Integration checks:

```bash
.venv-sglang/bin/python scripts/sglang/test_remote.py
python -m unittest discover -s tests -p test_sglang_benchmark.py -v
```

Local timing/telemetry and raw outputs live under `outputs/` and are excluded from
Git. Local timing and telemetry are not part of the public source export.

## Single-request anchor sweep

Keep the remote Target running and stop the existing Draft first. The sweep
starts a fresh Draft for each anchor count, waits for readiness, reuses the
baseline's workload, and stops only its own Draft process tree after each run:

```bash
unset MAX_BRANCHES
.venv-sglang/bin/python scripts/sglang/sweep_anchors.py \
  --anchors 1 2 3 4 6 \
  --dataset /path/to/math500-test.jsonl --tokenizer "$TARGET_MODEL" \
  --baseline outputs/sglang/ar.json --output-dir outputs/sglang/anchor-sweep
```

This sweep uses block size 16; keep the Target's `BLOCK_SIZE=16`. It expands the
branch limit with the anchor count and writes individual timing reports, Draft
logs, hit/miss telemetry, and an aggregate summary. Treat the selected value as
a workload-specific tuning result, not a universal optimum for SGLang.
