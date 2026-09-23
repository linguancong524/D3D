"""Smoke test v2: draft forward cost (CPU vs GPU), target verify cost, and
whether the async draft is hidden.

Three modes, all at anchor/micro fixed by the caller:

* measured -- draft timed on the main thread with non-blocking CUDA events; the
  engine's own deferred sync point is left untouched, so throughput matches the
  uninstrumented sweep.
* forced   -- same plus an explicit sync on cuda:1 at the end of the draft
  region. That sync is what the deferred path does anyway, just earlier: if
  throughput drops, the draft carried work that was previously overlapping the
  next round's submit + target verify.
* inject   -- real GPU work (warmed fp32 matmul) queued on the draft stream;
  throughput flat => that much draft GPU time is absorbable slack.

Target verify is timed by the target worker's own `profile_forward` spans
(sync on cuda:0 around runner.forward + posterior D2H).

Note: `torch.cuda._sleep` is unusable on this stack (1e7 cycles ~ 1 s, 1e8 is a
no-op), and an unwarmed inject-op calibration overstates per-op cost ~60x.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from bench import TARGET, DRAFT, original, AutoTokenizer, Engine


def stats(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return dict(n=len(ordered), mean=statistics.fmean(ordered),
                median=statistics.median(ordered), p90=ordered[int(0.9*(len(ordered)-1))],
                min=ordered[0], max=ordered[-1], total_ms=sum(ordered))


def calibrate_inject(device=1, warmup=20, repeats=30):
    """ms of real GPU work per injected op, measured after warmup."""
    scratch = torch.zeros(4096, 4096, device=device)
    out = torch.empty_like(scratch)
    for _ in range(warmup):
        torch.mm(scratch, scratch, out=out)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        torch.mm(scratch, scratch, out=out)
    torch.cuda.synchronize(device)
    return 1000*(time.perf_counter()-started)/repeats, (scratch, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', default=TARGET)
    parser.add_argument('--draft', default=DRAFT)
    parser.add_argument('--pages', type=int, default=512)
    parser.add_argument('--concurrencies', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--anchor', type=int, default=1)
    parser.add_argument('--anchors', type=int, nargs='+', default=None,
                        help='anchor list; overrides --anchor')
    parser.add_argument('--modes', nargs='+', default=['measured', 'forced', 'inject'],
                        choices=['measured', 'forced', 'inject'])
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--inject-ms', type=float, nargs='+', default=[20, 40, 80])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(TARGET)
    rows, cfg, _ = original.load_rows('aime25', 16)
    prompts = [tok.encode(original._apply_chat_template(tok, [{'role': 'user', 'content': cfg['question'](r)}], False)) for r in rows]

    op_ms, scratch = calibrate_inject(1)
    data = dict(dataset='aime25', anchor_num=args.anchor, requests=len(prompts),
                max_new_tokens=args.max_new_tokens, target=args.target, draft=args.draft,
                pages=args.pages, backup='neural',
                inject_op=f'4096x4096 fp32 matmul on cuda:1, {op_ms:.3f} ms/op', cells=[])
    print(f'calibrated inject op (warmed): {op_ms:.3f} ms/op', flush=True)
    engine = Engine(args.target, args.draft, pages=args.pages)

    cpu_spans, gpu_events, recover_spans, rounds = [], [], [], []
    inject_ops, forced = [0], [False]
    original_predict, original_recover = engine._predict, engine._recover
    original_submit, original_receive = engine.target.submit_forward, engine.target.receive

    def timed_predict(seqs):
        started = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_predict(seqs)
        for _ in range(inject_ops[0]):
            torch.mm(scratch[0], scratch[0], out=scratch[1])
        end.record()
        cpu_spans.append((started, time.perf_counter(), len(seqs)))
        gpu_events.append((begin, end))
        if forced[0]:
            torch.cuda.synchronize(1)
        return result

    def timed_recover(seqs):
        started = time.perf_counter()
        original_recover(seqs)
        recover_spans.append((started, time.perf_counter(), len(seqs)))

    awaiting = [False]

    def timed_submit(seqs, blocks):
        original_submit(seqs, blocks)
        rounds.append([time.perf_counter(), None, len(seqs)])
        awaiting[0] = True

    def timed_receive():
        result = original_receive()
        if awaiting[0]:
            rounds[-1][1] = time.perf_counter()
            awaiting[0] = False
        return result

    def run_cell(concurrency, kind, inject_ms=0.0, anchor=None):
        anchor = args.anchor if anchor is None else anchor
        engine.A, engine.micro = anchor, concurrency
        print(f'WARMUP C={concurrency} A={anchor} micro={concurrency} {kind}', flush=True)
        inject_ops[0], forced[0] = 0, False
        engine.generate(prompts[:concurrency], concurrency, max_new_tokens=32)
        for spans in (cpu_spans, gpu_events, recover_spans, rounds):
            spans.clear()
        engine._predict, engine._recover = timed_predict, timed_recover
        engine.target.submit_forward, engine.target.receive = timed_submit, timed_receive
        engine.target.request('profile_forward', True)
        forced[0] = kind == 'forced'
        inject_ops[0] = int(round(inject_ms/op_ms))
        print(f'RUN C={concurrency} {kind} inject={inject_ms}ms ops={inject_ops[0]}', flush=True)
        result = engine.generate(prompts, concurrency, max_new_tokens=args.max_new_tokens)
        spans = engine.target.request('forward_spans')
        engine.target.request('profile_forward', False)
        engine._predict, engine._recover = original_predict, original_recover
        engine.target.submit_forward, engine.target.receive = original_submit, original_receive
        inject_ops[0], forced[0] = 0, False
        torch.cuda.synchronize(1)

        verify = [1000*(e-s) for s, e in spans]
        cpu_ms = [1000*(e-s) for s, e, _ in cpu_spans]
        gpu_ms = [b.elapsed_time(e) for b, e in gpu_events]
        round_ms = [1000*(r[1]-r[0]) for r in rounds if r[1]]
        paired = list(zip(spans, cpu_spans))
        slack = [1000*(c[1]-t[1]) for t, c in paired]  # +ve: verify ended before draft region
        hits = sum(o['hits'] for o in result['outputs'])
        cell = dict(kind=kind, concurrency=concurrency, anchor_num=anchor, draft_microbatch=concurrency,
                    inject_ms=inject_ms, injected_ops=int(round(inject_ms/op_ms)),
                    tps=result['tps'], elapsed_s=result['elapsed_s'], rounds=len(spans),
                    verify_ms=stats(verify), draft_cpu_ms=stats(cpu_ms), draft_gpu_ms=stats(gpu_ms),
                    round_ms=stats(round_ms), verify_inside_draft_fraction=(
                        statistics.fmean(1.0 if x > 0 else 0.0 for x in slack) if slack else None),
                    recover_ms=stats([1000*(e-s) for s, e, _ in recover_spans]),
                    hit_rate=hits/max(1, hits+result['miss_requests']))
        data['cells'].append(cell)
        args.output.write_text(json.dumps(data, indent=2))
        print(f"{kind.upper()} C={concurrency} tps={cell['tps']:.1f} verify={cell['verify_ms']['mean']:.2f}ms "
              f"draft_cpu={cell['draft_cpu_ms']['mean']:.2f}ms draft_gpu={cell['draft_gpu_ms']['mean']:.2f}ms "
              f"round={cell['round_ms']['mean']:.2f}ms recover_n={cell['recover_ms']['n']}", flush=True)
        return cell

    anchor_list = args.anchors or [args.anchor]
    try:
        for concurrency in args.concurrencies:
            for anchor in anchor_list:
                for kind in args.modes:
                    if kind == 'inject':
                        for ms in args.inject_ms:
                            run_cell(concurrency, 'inject', inject_ms=ms, anchor=anchor)
                    else:
                        run_cell(concurrency, kind, anchor=anchor)
    finally:
        engine.close()
        args.output.with_suffix('.md').write_text(render(data))
        print(render(data), flush=True)


def render(data):
    lines = [f"# draft verify vs drafter smoke v2 (anchor={data['anchor_num']}, micro = concurrency, "
             f"{data['max_new_tokens']} new tokens, {data['inject_op']})", '',
             '| C | Mode | Injected ms | TPS | Verify mean ms | Draft CPU mean ms | Draft GPU mean ms | '
             'Round mean ms | Recover n | Recover mean ms | Hit rate |',
             '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in data['cells']:
        rec = c['recover_ms']
        lines.append(f"| {c['concurrency']} | {c['kind']} | {c['inject_ms']:.0f} | {c['tps']:.1f} | "
                     f"{c['verify_ms']['mean']:.2f} | {c['draft_cpu_ms']['mean']:.2f} | "
                     f"{c['draft_gpu_ms']['mean']:.2f} | {c['round_ms']['mean']:.2f} | {rec['n']} | "
                     f"{(rec['total_ms']/max(1,rec['n'])):.2f} | {c['hit_rate']:.1%} |")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    main()
