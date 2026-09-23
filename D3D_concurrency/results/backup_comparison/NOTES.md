# Interpretation and validation

The measured SDAR fast adaptation does not beat neural backup in this experiment.
At C=8/16/32 it is 8.0%/8.6%/9.6% slower. Neural remains the default.

Workload: first 64 Math500 prompts, max 256 output tokens, natural EOS, greedy,
two repetitions with reversed order, one model load, two H800 GPUs. All 16
measurements completed and passed the report's request, token, EOS, page-release,
timing-formula, backup-counter and workload-hash assertions. Five CPU tests pass.
Independent HF greedy validation passed for four 32-token prompts. SDAR recovery
and all 736 packed candidate tokens matched the original reference. Fast completed
40-request refill checks at C=8/16/32 before timing the comparison.

At C=32 neural has 7.33 actual output tokens per request-step, versus 6.64 for
fast. There are 184 versus 218 scheduler rounds across the two repetitions.
Thus lower useful progress is a directly measured cost. The implementation also
needs masked SDAR seeding for next-round anchor predictions; moving this work
after target submission does not eliminate it. No per-stage timing ablation was
performed here, so its exact share of the slowdown is not established.

Scope of the SSD reference (commit d7eb8fa0edb77a6d0876af1903367b9bb82f54e7):
`ssd/engine/draft_runner.py` initializes random-logit miss responses, returns
them, then performs AR glue-decode/tree speculation. `bench/bench.py --backup`
selects `fast` versus `jit`. Our greedy-only adapter generates uniform integer
tokens directly, retaining the verified anchor, and uses SDAR masked seeding
instead of AR next-token logits. These RNG distributions are not bitwise or
BF16-tie equivalent. The 46-branch topology is held fixed; SSD supports separate
miss fanout, which is not implemented here. Initial admission uses the same
neural bootstrap for both modes. This is a policy experiment in D3D, not a full
reproduction of the paper's optimized fast implementation.

Same-concurrency full-output agreement between neural and fast is 64/64 at C=1,
59/64 at C=8, 54/64 at C=16 and 57/64 at C=32 in each repetition. Different
speculation trajectories and batch composition may change BF16 greedy outputs;
first divergent logits were not independently diagnosed in this experiment.
No full Math500 accuracy claim is made. Reported TPS always uses actual output
tokens, and cross-policy output equality is not assumed.

Files: raw.json contains all timed cells, run.log the chronological log,
summary.json per-policy statistics and agreement counts, table.md the table.
Validation artifacts are ../backup_fast_validation.json and its .log.
