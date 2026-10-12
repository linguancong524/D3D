# Third-Party Notices

MIT applies to D3D-authored code in this repository. Copied or adapted
third-party code, model checkpoints, and benchmark datasets retain their own
licenses and terms.

## SSD backend

The files under `serving/vendor/ssd/` are selected from
[SSD](https://github.com/tanishqkumar/ssd), commit
`d7eb8fa0edb77a6d0876af1903367b9bb82f54e7`, under its MIT license. The
original notice is preserved in `serving/vendor/SSD_LICENSE`.

## Models and datasets

Model checkpoints and datasets are downloaded from their respective hosting
projects. Users must follow the licenses and terms shown by the upstream model
and dataset repositories.

## Qwen3-derived draft compatibility code

The files under `models/d3d/` retain Apache-2.0 attribution for the Qwen and
Hugging Face teams. The modeling file identifies Transformers v4.52.4
`models/qwen3/modeling_qwen3.py` as its basis. The Apache-2.0 license is
included at `models/d3d/APACHE_LICENSE`.

The draft implementation was imported through the SDAR code path in the
original experiment workspace and is now exposed under D3D module and class
names. Renaming does not change its provenance or grant additional rights.
Original attribution is preserved. The local adaptations include paged
attention integration, RoPE registry compatibility, and CPU-safe optional
kernel imports. Confirm the provenance of any additional SDAR-specific
adaptations before publication; a permissive license on the base Qwen code
does not determine the license of other upstream modifications.
