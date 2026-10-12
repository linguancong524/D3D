Selected MIT-licensed files from https://github.com/tanishqkumar/ssd
Commit: d7eb8fa0edb77a6d0876af1903367b9bb82f54e7
Original license: SSD_LICENSE.
Qwen3 model, linear/norm/rotary/embedding layers, weight loader, Sequence,
and BlockManager reused. Context and attention are local replacements.
Local engine uses strict request admission and D3D parallel block drafting;
it does not use SSD's autoregressive draft runner or original benchmark.
