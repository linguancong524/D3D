# Fast Backup diagnosis

Pooled actual output tokens / wall time including prefill, natural EOS. Ratios compare backup policies, not AR speedup. SDAR-1.7B -> Qwen3-8B.

| Dataset | Cap | Samples | Anchor | C | Policy | TPS | vs Neural | Tokens/request-step | ms/request-step | Misses |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| math500 | 128 | 32 | 3 | 16 | neural | 303.9 | 1.000 | 6.66 | 21.91 | 188 |
| math500 | 128 | 32 | 3 | 16 | fast_legacy | 232.9 | 0.766 | 6.18 | 26.53 | 174 |
| math500 | 128 | 32 | 3 | 16 | fast(miss=3) | 228.4 | 0.751 | 6.18 | 27.05 | 174 |
| math500 | 128 | 32 | 3 | 32 | neural | 294.6 | 1.000 | 6.61 | 22.42 | 188 |
| math500 | 128 | 32 | 3 | 32 | fast_legacy | 277.7 | 0.942 | 6.08 | 21.89 | 174 |
| math500 | 128 | 32 | 3 | 32 | fast(miss=3) | 312.5 | 1.061 | 6.08 | 19.45 | 174 |
| math500 | 256 | 32 | 1 | 16 | neural | 509.4 | 1.000 | 6.98 | 13.71 | 1278 |
| math500 | 256 | 32 | 1 | 16 | fast(miss=1) | 422.8 | 0.830 | 5.50 | 13.01 | 1326 |
| math500 | 256 | 32 | 1 | 32 | neural | 572.1 | 1.000 | 6.90 | 12.05 | 1299 |
| math500 | 256 | 32 | 1 | 32 | fast(miss=1) | 508.8 | 0.889 | 5.46 | 10.73 | 1344 |
| math500 | 256 | 32 | 1 | 32 | neural | 662.0 | 1.000 | 6.90 | 10.42 | 1299 |
| math500 | 256 | 32 | 1 | 32 | fast(miss=16) | 541.6 | 0.818 | 5.65 | 10.42 | 1149 |

Source: `D3D_concurrency/results/fast_v2_a3.json`; repeats=2; fast miss anchors=0 (0 means same as anchor).
Source: `D3D_concurrency/results/fast_v2_a1.json`; repeats=3; fast miss anchors=0 (0 means same as anchor).
Source: `D3D_concurrency/results/fast_v2_wide.json`; repeats=3; fast miss anchors=16 (0 means same as anchor).
