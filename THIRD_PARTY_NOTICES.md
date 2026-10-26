# Third-Party Notices

MIT applies to D3D-authored code in this repository. Copied or adapted
third-party code, model checkpoints, and benchmark datasets retain their own
licenses and terms.

## SSD backend

The files under `serving/vendor/ssd/` are selected from
[SSD](https://github.com/tanishqkumar/ssd), commit
`d7eb8fa0edb77a6d0876af1903367b9bb82f54e7`, under its MIT license. The
original notice is preserved in `serving/vendor/SSD_LICENSE`.

## Qwen3-derived draft compatibility code

The files under `models/d3d/` and `models/d3d_training/` retain Apache-2.0 attribution for the Qwen and
Hugging Face teams. The modeling file identifies Transformers v4.52.4
`models/qwen3/modeling_qwen3.py` as its basis. The Apache-2.0 license is
included at `models/d3d/APACHE_LICENSE`.

The training and progressive-pruning implementation is integrated from the
D3D authors' `d3d_release` training release. Its training-specific model port
is kept in `models/d3d_training/` to preserve the existing inference runtime.
See [the training guide](docs/training.md) for the shared native environment
and separate optional SGLang runtime.

## SDAR

The released draft checkpoints use the SDAR checkpoint layout
(`model_type: sdar`, `SDARForCausalLM`), and the local drafter was imported
through that code path before being exposed under D3D module names. Upstream:
[SDAR (Synergy of Diffusion and AutoRegression)](https://github.com/JetAstra/SDAR),
MIT license; consult the upstream repository and model cards for
checkpoint-specific terms.

## SGLang backend

`third_party/sglang/` is a source snapshot of SGLang and the decoupled
implementation contributed through `wintem0531/sglang-speedup-nvidia`, commit
`db686d124840ebfdd61e787b3a7fbf543244b6f0`. Its SGLang baseline is `v0.5.18`,
commit `71de97b264b04dcd514cf904003028aefe9775c8`. Original license files are
preserved, including `third_party/sglang/LICENSE`; those sources are not
relicensed under D3D's MIT license. See `third_party/sglang/UPSTREAM_D3D.md`
for attribution and export exclusions. D3D's default Hybrid launcher wraps
the fork's original `DECOUPLED` implementation; the optional strict-remote
plugin is retained in `d3d_sglang/`.

## Model and dataset terms

Model checkpoints and datasets are downloaded from their respective hosting
projects. Users must follow the licenses and terms shown by the upstream model
and dataset repositories.
