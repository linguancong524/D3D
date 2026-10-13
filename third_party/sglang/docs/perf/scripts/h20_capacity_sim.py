#!/usr/bin/env python3
"""H20 deployment capacity sim: Qwen3.5-9B bf16, TP=1 replicas x N, 100 concurrent users (512 in / 256 out)."""
import math, statistics as st

WEIGHTS_GB = 17.62
BW_EFF_GBMS = 4.0 * 0.73          # H20 4.0TB/s, assume same 73% effective as H800 measured -> 2.92 GB/ms
KV_PER_TOK_GB = 30.52e-6
MAMBA_PER_REQ_GB = 0.115
COMP_TFLOPS = 148 * 0.35          # decode compute assumption 35% MFU
FLOP_PER_TOK = 15e9

def decode_step_ms(d, ctx, h20=True):
    if h20:
        mem = (WEIGHTS_GB + d * (MAMBA_PER_REQ_GB + ctx * KV_PER_TOK_GB)) / BW_EFF_GBMS
        comp = d * FLOP_PER_TOK / (COMP_TFLOPS * 1e9)  # ms
        return max(mem, comp)
    else:  # H800 anchor model
        return 7.17 + d * (48 + (ctx - 128) * 0.0109) / 1000.0

SMALL_CHUNK, LARGE_CHUNK = 512, 8192
def make_prefill_cost(small_us_tok, large_us_tok):
    ls, ll = math.log(small_us_tok), math.log(large_us_tok)
    lc1, lc2 = math.log(SMALL_CHUNK), math.log(LARGE_CHUNK)
    def cost(chunk):
        if chunk <= SMALL_CHUNK: return small_us_tok
        if chunk >= LARGE_CHUNK: return large_us_tok
        return math.exp(ls + (ll - ls) * (math.log(chunk) - lc1) / (lc2 - lc1))
    return cost

def ctx_avg(dec):
    if not dec: return 128
    return sum(it[2] for it in dec) / len(dec)

def simulate_card(initial_reqs, prefill_cost, chunk_cap, t_max=120000.0, id0=0):  # t in ms
    dec = []          # [req, n, ctx, last_tok_time]
    pqueue = []
    pending = []
    t = 0.0
    out = {}
    nid = [id0]
    for r in initial_reqs:
        pending.append((0.0, r))
    it = 0
    while it < 400000:
        it += 1
        # absorb arrivals
        if pending:
            pend = []
            for a in sorted(pending, key=lambda x: x[0]):
                if a[0] <= t:
                    pqueue.append(a[1])
                else:
                    pend.append(a)
            pending = pend
        d = len(dec)
        if d == 0 and not pqueue and not pending:
            break
        if pqueue:
            chunk_reqs, toks = [], 0
            while pqueue and toks < chunk_cap:
                r = pqueue.pop(0)
                chunk_reqs.append(r); toks += r["in_len"]
            T = toks * prefill_cost(toks) / 1e3 + (decode_step_ms(d, ctx_avg(dec)) if d else 0.0)  # ms
            t += T
            for r in chunk_reqs:
                out[r["id"]] = {"ttft": t - r["arr"], "itls": [], "arr": r["arr"]}
                dec.append([r, 0, r["in_len"], t])
        else:
            T = decode_step_ms(d, ctx_avg(dec))
            t += T
            for item in dec:
                r, n, c, last = item
                n += 1; c += 1; item[1] = n; item[2] = c; item[3] = t
                out[r["id"]]["itls"].append(t - last)
            done = [it2 for it2 in dec if it2[1] >= it2[0]["out_len"]]
            for it2 in done:
                r = it2[0]
                out[r["id"]]["e2e"] = t - r["arr"]
                r2 = dict(r); r2["id"] = nid[0]; nid[0] += 1; r2["arr"] = t
                pending.append((t, r2))   # sustained: same user resubmits immediately
            dec = [it2 for it2 in dec if it2[1] < it2[0]["out_len"]]
        if t > t_max:
            break
    return out

def run(N, users=100, in_len=512, out_len=256, small_us=115.0, large_us=197.5, chunk_cap=8192):
    prefill_cost = make_prefill_cost(small_us, large_us)
    per_card = {c: [] for c in range(N)}
    for i in range(users):
        r = {"id": i, "arr": 0.0, "in_len": in_len, "out_len": out_len}
        per_card[i % N].append(r)
    all_out = {}
    for c in range(N):
        res, _ = None, None
        out_c = simulate_card(per_card[c], prefill_cost, chunk_cap)
        all_out.update(out_c)
    ttfts = [r["ttft"] for r in all_out.values()]
    itls = [x for r in all_out.values() for x in r["itls"]]
    e2es = [r["e2e"] for r in all_out.values() if "e2e" in r]
    dec_rates = [1000.0 / st.mean(r["itls"]) for r in all_out.values() if len(r["itls"]) >= 2]
    return {"N": N, "ttft_p50": pctl(ttfts,50), "ttft_p90": pctl(ttfts,90), "ttft_p99": pctl(ttfts,99),
            "itl_p50": pctl(itls,50), "itl_p90": pctl(itls,90), "itl_p99": pctl(itls,99), "itl_max": max(itls),
            "itl_mean": st.mean(itls), "e2e_p90": pctl(e2es,90), "e2e_p99": pctl(e2es,99),
            "dec_p50": pctl(dec_rates,50), "dec_min": min(dec_rates)}

def pctl(v, p):
    if not v: return float("nan")
    v = sorted(v); k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)

BASE_SMALL = 115.0
BASE_LARGE = 29.4 * (990 / 148)

def show(scen, label, configs):
    print(f"\n===== {label} =====")
    print(f"{'N':>3} | {'TTFT p50/p90/p99 (ms)':<24} | {'ITL p50/p90/p99/max (ms)':<28} | {'E2E p90':>8} | {'tok/s p50/min':>14}")
    for N, kw in configs:
        r = run(N, **kw)
        print(f"{r['N']:>3} | {r['ttft_p50']:6.0f}/{r['ttft_p90']:6.0f}/{r['ttft_p99']:6.0f}"
              f"{'':13} | {r['itl_p50']:6.1f}/{r['itl_p90']:6.1f}/{r['itl_p99']:6.1f}/{r['itl_max']:7.1f}"
              f"{'':11} | {r['e2e_p90']:8.0f} | {r['dec_p50']:8.0f}/{r['dec_min']:6.0f}")

show("s1", "S1 optimistic: small prefill bandwidth-limited (k=0.84), large chunk 197us/tok (4.9k tok/s), cap 8192",
     [(N, dict(small_us=BASE_SMALL*0.84, large_us=BASE_LARGE, chunk_cap=8192)) for N in [3,4,5,6,8,10]])
show("s2", "S2 baseline: small prefill k=1.5 (~170us/tok, 87ms/512tok), large chunk 197us/tok, cap 8192",
     [(N, dict(small_us=BASE_SMALL*1.5, large_us=BASE_LARGE, chunk_cap=8192)) for N in [3,4,5,6,8,10]])
show("s3", "S3 pessimistic: small prefill k=5 (~295ms/512tok), large chunk 250us/tok (4k tok/s), cap 8192",
     [(N, dict(small_us=BASE_SMALL*5, large_us=250.0, chunk_cap=8192)) for N in [4,5,6,8,10,12]])
show("s4", "S4 pessimistic prefill + chunk_cap 2048 (mitigate ITL spikes)",
     [(N, dict(small_us=BASE_SMALL*5, large_us=250.0, chunk_cap=2048)) for N in [6,8,10,12]])

print("===== S5 optimistic compute: large chunk 150us/tok (6.7k tok/s), small 87ms, cap 8192 =====")
show("s5", "S5 optimistic-compute: large 150us/tok, small 87ms",
     [(N, dict(small_us=172.5, large_us=150.0, chunk_cap=8192)) for N in [3,4,5,6,8]])
print()
print("===== scale-out: 200 concurrent users (S2 baseline / S3 pessimistic) =====")
show("s2b", "S2 baseline @200 users", [(N, dict(small_us=172.5, large_us=196.66, chunk_cap=8192, users=200)) for N in [8,10,12,14,16]])
show("s3b", "S3 pessimistic @200 users", [(N, dict(small_us=575.0, large_us=250.0, chunk_cap=8192, users=200)) for N in [10,12,14,16,20]])
print("\n===== model validation (H800 anchors) =====")
print(f"H800 single 512-tok prefill est: {512*115/1e3:.0f} ms  (measured serving TTFT median 70.5ms incl ~5ms overhead)")
print(f"H800 decode S(13,640) = {decode_step_ms(13,640,h20=False):.2f} ms (measured rate-4 serving TPOT median 8.33ms)")
print(f"H20 decode S(25,640) = {decode_step_ms(25,640):.2f} ms -> {1000/decode_step_ms(25,640):.0f} tok/s per user")
print(f"H20 decode S(50,640) = {decode_step_ms(50,640):.2f} ms -> {1000/decode_step_ms(50,640):.0f} tok/s per user")
print(f"H20 decode S(100,640) = {decode_step_ms(100,640):.2f} ms -> {1000/decode_step_ms(100,640):.0f} tok/s per user")
print(f"H20 large-chunk prefill: {1e6/BASE_LARGE:.0f} tok/s (base) ; 4000 tok/s (pessimistic)")
