# Vendored SGLang provenance

- Source: `https://github.com/wintem0531/sglang-speedup-nvidia`
- Source commit: `db686d124840ebfdd61e787b3a7fbf543244b6f0`
- Upstream: `https://github.com/sgl-project/sglang`
- Upstream baseline: annotated tag `v0.5.18`, peeled commit
  `71de97b264b04dcd514cf904003028aefe9775c8`.
- Imported on: 2026-10-13.
- Preserved: original source, license files, setup guides, tests, benchmark
  scripts, configurations and reports.
- Omitted: `docs/perf/results`, `docs/perf/logs`, `docs/perf/profiles`, Git
  metadata and fork-specific agent instructions (`AGENTS.md`, `CLAUDE.md`,
  `.claude`). Reports referring to omitted evidence are upstream reports,
  not newly reproduced D3D results.
- D3D integration lives outside this snapshot in `d3d_sglang/`. No original
  source file was changed during import. The default launcher wraps the fork's
  Hybrid Target-local required Draft and remote optional Draft path. The
  opt-in strict-remote plugin uses SGLang's plugin registration API and its
  host-proposal verifier without a Target-local Draft.

The original SGLang license is retained in `LICENSE`. Fork contributions are
attributed to the source repository and its contributors; retaining this
snapshot does not relicense them as D3D-authored MIT code.
