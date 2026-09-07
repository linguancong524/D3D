anchor-k


draft权重在:
(qwen3-8B)
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-8-a128-v2/step-340000   
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-16-a128-ar1.0/step-355227
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/block-32/anchor64-ar1.0/step-355227
分别是sdar-8b b8/16/32的路径

/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar02_a256_flash/step-355227              
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/baseline_b32_a80_ar1.0_from355227_ce_2epoch_4gpu_fixedcode/step-360000                      
分别是sdar-1.7b  b8/16/32的路径

/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sampling/sdar17b_b16_a128_ar1.0/step-1065681
sampling的1.7bdrafter 

/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge/APLF/results/progressive/b16_progressive_seed42/d31/train/step-16000
是sdar-8b切31层 b16的路径

/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/greedy/sdar4b/block-16-a128-ar1.0/step-355227
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sampling/sdar4b_b16_a128_ar1.0/step-350000
是sdar-4B b16 greedy和sampling路径

(qwen3-32B)
/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16_a128_ar1.0_from255576_ce_3epoch_4gpu/resume_from_step170000/resume_from_step140000/step-731183
是sdar-8B b16 greedy路径

/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar17b_b16_a128_ar1.0_from255044_ce_3epoch_4gpu/step-1041183 
是sdar-1.7B b16 greedy路径





现在的异步相当于一种hint策略，一次verify结束后，会返回真实的截断位置和真实锚点，和我们在verify期间进行的异步draft结果进行比对，hint中我们其中一种draft结果，所以acclen和hint命中率(没有命中会导致回滚)共同决定了speedup。对于异步draft过程，我们引入budget来度量预测的成本，budget指的是dllm在此次draft前向中unmask的token个数。增加budget可以提高hint命中率，但是消耗更多的算力甚至拖慢整体speedup：在一卡场景，增加budget同时拖慢verify和draft前向时间，在两卡场景主要拖慢异步draft前向时间使其不能被verify时间所覆盖。

所以可能的优化目标是消耗尽可能少的budget提高hint命中率。

现有方案对budget的消耗是 budget = k*n*block_size,其中
参数k(对某个截断位置取topk的token当锚点)，参数n(block内取n个可能的截断位置)

实验
1.7bq8bb16
8bq32bb16
8bq8bb16
8bq8bb32

取5*5的方格画三维图，轴分别是k,n,speedup
因为top1是默认被verify拒绝的，所以k取2,3,4,5,6,  n取2,4,8,12,16(block内前n-1个位置和block内最后一个位置)





异步跑法示例（目前最快的设置）:

# 8b对32b
CUDA_VISIBLE_DEVICES=0,1 \
MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
DATASETS=math500 \
BLOCK_SIZE=16 \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/qwen3-32b/sdar8b_b16_a128_ar1.0_from255576_ce_3epoch_4gpu/step-170000 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_smoke/8bq32bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 3 --packed-accepted-counts all

# 1.7b对32b num 7

# 1.7b对8b
CUDA_VISIBLE_DEVICES=0,1 \
MAX_SAMPLES=30 MAX_NEW_TOKENS=2048 \
DATASETS=math500 \
BLOCK_SIZE=16 \
VERIFIER=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-8B \
DRAFTER=/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908 \
OUT_DIR=/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/results_smoke/1.7bq8bb16  \
bash /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3DV2/scripts/eval_async_packed.sh --anchor-num 4 --packed-accepted-counts all

## Async anchor and sampling semantics

The async packed runner treats the first token of every verification block as a
target-produced anchor. The verifier forwards the complete block so the anchor
is written to target KV and conditions the following logits, but acceptance is
computed only for `block[:, 1:]`. The committed length is therefore one anchor
plus the accepted draft-tail length.

For temperature sampling, every recovery and packed candidate carries the exact
sparse proposal distribution used to sample each token. Its support width is
`proposal_top_k`, so temperature sampling requires `proposal_top_k > 0`. The
draft process sends only the support token IDs and normalized probabilities to
the verifier process. Tail tokens use standard `p/q` rejection sampling and
residual recovery. Unanchored packed outcomes, including full-pass outcomes,
overwrite their first token with the verifier's real correction or bonus
anchor; that overwritten position is never included in the next round's `p/q`
verification. The retained tail continues to use its original unanchored
proposal probabilities.
