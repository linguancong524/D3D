# Shared native training and inference environment

## Supported dependency set

`requirements.txt` is the single dependency source for offline training,
progressive pruning, native single-request inference, and native concurrent
inference.

| Package | Version |
|:---|:---|
| PyTorch | 2.8.0 (GPU checks: 2.8.0+cu128) |
| TorchVision | 0.23.0 |
| Transformers | 4.57.3 |
| Accelerate | 1.12.0 |
| TorchAO | 0.13.0 |
| FlashAttention | 2.8.3, installed separately after PyTorch |
| Datasets | 4.4.2 |
| Safetensors | 0.7.0 |

The separate training model implementation remains in `models/d3d_training/`.
Environment unification does not merge or change model arithmetic.

## GPU compatibility checks — 2026-10-26

Hardware: two NVIDIA H800 80 GB GPUs. Python: 3.12. Target:
`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
Draft: `lingc8/D3D-1.7B-to-8B`, revision
`23d559e19b110774c5930b077428b955ff020d8b`.

- Single-request asynchronous packed decoding: first three MATH-500 samples,
  128 output tokens per sample, block size 16, three anchors, BF16, SDPA.
  All three requests completed, producing 384 output tokens.
- Persistent shared `.venv`, with every entry in `requirements.txt` installed:
  single-request decoding also completed with FlashAttention-2 enabled on both
  target and draft, producing 128 tokens for one MATH-500 sample.
- Native concurrent decoding: first four MATH-500 samples, 64 output tokens
  per sample, concurrency 1 and 4, block size 16, three anchors, 128 KV pages.
  Both cells completed, each producing 256 output tokens. This path exercises
  FlashAttention and the paged-KV backend.
- Target-reference validation: greedy outputs exactly matched independent
  Transformers target decoding for all four prompts, 32 tokens per prompt.
- Draft-reference validation: recovery matched; all 46 packed candidate keys
  matched, with 736/736 candidate tokens equal to the reference implementation.

These are compatibility smoke tests, not a new speed benchmark, complete
MATH-500 accuracy evaluation, or proof of compatibility for every checkpoint.
The baseline and speculative test outputs are truncated intentionally.
The final CPU regression suite ran 52 tests successfully, with the GPU-only
Liger check skipped because CUDA was hidden for that run.

## Repeat checks

Activate the shared `.venv` and run the CPU suite, including tiny packed-training
forward/backward, physical layer pruning, and loading a training checkpoint
through the native inference loader:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

For a two-GPU reference check with the released checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m serving.bench \
    --target Qwen/Qwen3-8B --draft lingc8/D3D-1.7B-to-8B \
    --datasets math500 --max-samples 4 --max-new-tokens 64 \
    --concurrency 1 4 --anchor-num 3 --pages 128 --validate \
    --output outputs/unified_environment/concurrent.json
```

Use the exact revisions above, or local snapshots, when reproducing the recorded
check. The entry point records runtime metadata and reference comparisons.

## GPU training checks — 2026-10-31

The same persistent `.venv` used for native evaluation also completed two-GPU
FSDP2 training with the released 1.7B draft and Qwen3-8B tokenizer revisions
listed above. Both SDPA and FlashAttention-2 completed two optimizer updates
with BF16, block size 16, four anchors per example, CE plus AR loss, and no
gradient accumulation. The eight synthetic arithmetic conversations were used
only to check the training pipeline, with sequence limits of 512 tokens.

SDPA block loss was 5.490164 then 5.023682; AR loss was 2.831461 then 2.683333.
Its step-2 checkpoint was saved successfully. FlashAttention-2 block loss was
5.495502 then 5.023467; AR loss was 2.831461 then 2.677778. Both runs supervised
112 and 96 block tokens across the two steps. These checks establish working
forward/backward and optimizer execution, not convergence or training quality.
The saved SDPA checkpoint also loaded through the native asynchronous packed
evaluation entry point and completed one MATH-500 request with 32 output tokens.

`requirements.txt` is the only native dependency entry point. The obsolete
training requirements alias has been removed. SGLang continues to use its
separate environment below.

## SGLang remains separate

The optional fork currently pins PyTorch 2.13.0 and Transformers 5.12.1 in
`third_party/sglang/python/pyproject.toml`, and its own CUDA kernels in
`scripts/sglang/runtime.lock`. Those pins conflict with this training stack.
Keep `.venv-sglang` and its installation script unchanged. Unifying that stack
would require a separate port and GPU validation, not just changing a requirements
file. See [the SGLang guide](sglang-backend.md).
