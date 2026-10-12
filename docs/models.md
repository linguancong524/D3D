# Released draft checkpoints

All six checkpoints are grouped in the public
[lingc8 / D3D collection](https://huggingface.co/collections/lingc8/d3d-6acc999de5b699dcf37b3c75).
This collection groups existing repositories; it does not move or rename
them, and their model IDs remain valid.

The six release checkpoints are the exact selections in the original release
list.

| Hub repository | Target | Layers | Tensor parameters | Weight bytes |
|:---|:---|---:|---:|---:|
| [D3D-1.7B-to-8B](https://huggingface.co/lingc8/D3D-1.7B-to-8B) | Qwen3-8B | 28 | 2,031,739,904 | 8,126,995,168 |
| [D3D-1.7B-to-32B](https://huggingface.co/lingc8/D3D-1.7B-to-32B) | Qwen3-32B | 28 | 2,031,739,904 | 8,126,995,168 |
| [D3D-8B-to-8B](https://huggingface.co/lingc8/D3D-8B-to-8B) | Qwen3-8B | 36 | 8,190,735,360 | 32,762,987,240 |
| [D3D-8B-to-32B](https://huggingface.co/lingc8/D3D-8B-to-32B) | Qwen3-32B | 36 | 8,190,735,360 | 32,762,987,240 |
| [D3D-24D-p1.7B-to-32B](https://huggingface.co/lingc8/D3D-24D-p1.7B-to-32B) | Qwen3-32B | 24 | 1,830,395,904 | 7,321,613,968 |
| [D3D-31D-p8B-to-8B](https://huggingface.co/lingc8/D3D-31D-p8B-to-8B) | Qwen3-8B | 31 | 7,226,003,200 | 28,904,052,320 |

Parameter counts sum the tensor shapes in the safetensors headers, including
embedding and output tensors. Names describe the base architecture family, not
the exact number of stored parameters. Byte counts include safetensors headers
and refer to the original FP32 files, not BF16 GPU residency.

## What is preserved

- Original safetensors shards and their complete weight index.
- Original tensor values and FP32 precision; no re-quantization or dtype conversion.
- Tokenizer and generation assets present in the source checkpoint.
- Architecture dimensions and retained-layer count.

## Packaging for inference

The release adds model cards, license text, and `release_metadata.json`.
The original Hub packages include draft compatibility files and an inference
`AutoModelForCausalLM` mapping rather than the training-time FSDP wrapper.
Those published files retain their original names and type tags. The current
GitHub implementation exposes `D3DConfig` and `D3DForCausalLM` under
`models/d3d/`, and accepts both D3D and legacy checkpoint type tags.
This naming cleanup changes neither tensor keys nor weights; it does not
rewrite the existing Hub repositories.

The supported entry points are the D3D scripts. They resolve the Hub snapshot
and load it through local compatibility classes with remote code execution
disabled. Hub-hosted Python is included for architecture inspection and
compatibility; arbitrary Transformers versions are not supported.

Training-time FSDP implementation files, optimizer state, recovery manifests,
evaluation logs, and local datasets are not uploaded.

## Validation boundaries

Publication checks cover public repository access, complete shard lists,
weight sizes and hashes, configuration dimensions, and tokenizer loading.
They do not establish generation quality, target-reference equivalence, or
performance for all six released checkpoints. Local GPU smoke and reference
checks have a narrower scope.

To reproduce a result, record the Hub commit SHA in addition to the model ID,
and use the same dataset samples, token cap, attention backend, and decoding
configuration.

## Maintain the collection

Authenticate with a local write-scoped Hub credential, then run:

```bash
python scripts/organize_hub.py --dry-run
python scripts/organize_hub.py --output docs/huggingface-collection.json
```

The helper reuses an existing D3D collection, checks that all six repositories
are public, and adds missing items without duplicates or weight uploads.
The collection's actual slug and model IDs are recorded in
`huggingface-collection.json`. A Hub collection lives under `/collections/`;
`lingc8/D3D` here describes the owner and collection title, not a combined
weight repository.
