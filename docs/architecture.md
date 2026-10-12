# D3D architecture

**D3D is the draft-and-verify execution architecture. Qwen3 supplies the
transformer backbone.** The draft runs with diffusion-style masked blocks;
the target retains its autoregressive verification role. Sharing a backbone
does not make standard Qwen3 generation a substitute for D3D drafting.

<p align="center">
  <img src="figure1.png" alt="D3D method overview: drafter training, progressive layer pruning, and asynchronous inference" width="960">
</p>

## Three cooperating components

| Component | Responsibility | Implementation |
|:---|:---|:---|
| Diffusion draft · GPU 1 | Generate blocks and packed continuation candidates | `models/d3d/`, `single/eval_async_packed.py` |
| Target verifier · GPU 0 | Verify pending drafts and provide correction/anchor tokens | `single/eval_sync.py`, `serving/backend.py` |
| Coordinator | Commit accepted tokens and select reusable candidates using target feedback | `single/eval_async_packed.py`, `serving/engine.py` |

## Decode cycle

1. Prefill the prompt and obtain an anchor token from the target.
2. Generate a masked draft block conditioned on that anchor.
3. Submit the pending block to target verification. While verification runs,
   prepare packed candidates for possible acceptance positions and anchors.
4. Commit the accepted prefix and target-selected token, then select a matching
   candidate. Recover a draft when the required candidate is unavailable.
5. Repeat until the stopping condition or output-token limit is reached.

The two GPUs perform independent work; verification feedback decides what can
be reused. The architecture diagram is schematic, not a measured timeline.

## One architecture, two execution paths

| Path | Entry point | Cache and scheduling |
|:---|:---|:---|
| Single request | `single/eval_async_packed.py` | Per-request draft and target caches |
| Concurrent requests | `python -m serving.bench` | Paged KV caches, strict request admission, draft microbatches |

`models/d3d/configuration_d3d.py` and `modeling_d3d.py` expose `D3DConfig`,
`D3DModel`, and `D3DForCausalLM`. Transformer parameter names and arithmetic
are unchanged by the public naming cleanup. The local loader accepts both
the new D3D type and legacy draft type tags, without executing Hub-hosted Python.

## What this release does not claim

The repository does not rebrand Qwen3 as an originally invented transformer
backbone. Upstream source notices remain in the model files and
[third-party notices](../THIRD_PARTY_NOTICES.md). Published checkpoints are
drafts paired with specific target models, not standalone target replacements.
No performance number follows from the diagram or from renaming the code;
see the [benchmark protocol](../benchmarks/README.md).
