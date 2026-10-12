# Release checklist

The repository is a clean source export, not an export of the surrounding
experiment workspace. Publication uses an isolated clone of
`linguancong524/D3D`; the enclosing local experiment repository is not pushed.
Existing GitHub legacy directories are preserved.

## Publication packaging — 2026-10-12

- The six selected source checkpoints and exact tensor counts are documented
  in [models.md](models.md).
- Weights retain their original FP32 precision. Training artifacts and caches
  are excluded; inference compatibility files and model cards are included.
- GitHub includes the current source, an architecture banner, model links,
  and CPU CI. No new benchmark-performance claim is made.
- Software citation metadata uses the collective name already present in
  the source license. Paper authors and affiliations are not fabricated.

## Before publication

- Use the paper's approved author/affiliation metadata for a research citation.
- Confirm the copyright holder for the MIT license.
- Confirm the source and redistribution terms of SDAR-specific adaptations;
  the preserved Apache and MIT notices cover their identified base code.
- Upload all six draft checkpoints to the listed Hugging Face IDs. Each
  repository needs config, tokenizer assets, complete weights, a short model
  card, architecture/pruning details, and the applicable model license.
- Verify anonymous downloads of the public checkpoints in a fresh cache.
- Run both quick starts on two H800 GPUs in a clean Python 3.10 environment.
- Run 2048-token benchmarks and the optional greedy-reference checks. Validate
  every target/draft pair; do not infer correctness from imports alone.
- The optional concurrent `--validate` currently loads a second target and a
  second draft alongside the serving engine. Avoid it for 32B or other models
  that cannot fit twice on each GPU; run independent, staged reference checks
  before publishing comparisons.
- Add the paper's approved result table/figure with matching run metadata.
- Revoke tokens previously shared in chat; authenticate locally with newly
  issued, least-privilege credentials. Do not share credentials in chat.
- Review the existing GitHub branch and preserve a backup before replacing
  its contents. Export only this directory, never the parent repository.

## Lightweight checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q single serving models d3d_utils.py
for script in scripts/*.sh; do bash -n "$script"; done
```

After installation, `bash scripts/smoke_test.sh` checks imports, the approved
PyTorch/Transformers versions, and visible CUDA devices. It does not download
weights or validate full-size inference.

## Organization-time validation

The local checks ran without a GPU: 16 unit tests passed, Python 3.10 syntax
was parsed, all shell entry points passed `bash -n`, and the CLI help commands
loaded under Python 3.11, PyTorch 2.6.0+cu124, and Transformers 4.52.4. The
cross-directory quick start failed early with an actionable CUDA error. This
is not a fresh Python 3.10 installation test. Anonymous Hugging Face
availability, H800 execution, and full benchmark results remain release gates.

## GPU follow-up

See `docs/gpu-validation.md` for the 2026-10-10 H800 follow-up. Local 8B/1.7B
single-request, concurrent, and reference checks passed after fixing optional
Liger device handling. A local 32B/24-layer draft pair also passed single-request
and four-request concurrent smoke checks using 128 KV pages. Public checkpoint
downloads, the remaining pairs,
and full benchmark results are still release gates.
