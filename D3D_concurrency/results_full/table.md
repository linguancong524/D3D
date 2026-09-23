Concurrency	1	4	8	16

1.7B -> 8B (Qwen3-8B)
HumanEval
192 (4.43x)	459 (2.9x)	611 (2.3x)	686 (1.4x)
Math500
248 (5.8x)	581 (4.1x)	742 (2.8x)	859 (1.8x)
MT-Bench
98 (2.6x)	221 (1.4x)	357 (1.3x)	420 (1.0x)

1.7B -> 32B (Qwen3-32B)
HumanEval
129 (5.1x)	337 (3.6x)	611 (3.6x)	659 (2.3x)
Math500
151 (6.8x)	420 (4.3x)	702 (3.7x)	795 (2.4x)
MT-Bench
67 (2.8x)	185 (1.9x)	290 (1.6x)	386 (1.4x)

Cells are the high value (fastest attempt) of the cell; the value in parentheses is that
TPS divided by the AR baseline TPS of the same verifier at the same concurrency (8B: pooled
measurement; 32B: quiet-node estimate, `results_baseline/32B/summary_n32_t512.csv`).
Still first-pass numbers, measured while the host was contended: 1.7b_8b humaneval; 1.7b_8b mt-bench; 1.7b_32b humaneval; 1.7b_32b math500; 1.7b_32b mt-bench.
Baseline rows carrying tps_source=estimated/substituted (not measured for that dataset): 1.7b_32b humaneval; 1.7b_32b math500; 1.7b_32b mt-bench.
