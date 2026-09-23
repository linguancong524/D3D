"""Smoke test: per-round target-verify vs drafter-forward cost at full microbatch,
and whether the async draft is hidden behind the target verify.

Target side uses the worker's built-in `profile_forward` spans (it syncs GPU0
before/after runner.forward and the posterior D2H). Draft side is timed in the
main process with an explicit sync on cuda:1 at both ends of the draft region,
which only moves where the existing deferred sync happens, not the work.

The hide test injects real GPU work into the draft stream: if the draft is
hidden, throughput stays flat until the injected delay exceeds the verify/draft
slack, then falls off. (`torch.cuda._sleep` is unusable here: 1e7 cycles runs
~1 s while 1e8 is a no-op on this stack.)
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
                min=ordered[0], max=ordered[-1])


INJECT_SHAPE = (2048, 2048)


def calibrate_inject(device=1, repeats=50):
    """Milliseconds of draft-stream GPU work per dummy add_ op."""
    scratch = torch.zeros(*INJECT_SHAPE, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        scratch.add_(1.0)
    torch.cuda.synchronize(device)
    per_op_ms = 1000*(time.perf_counter()-started)/repeats
    del scratch
    torch.cuda.empty_cache()
    return per_op_ms


def inject_delay(scratch, ops):
    for _ in range(ops):
        scratch.add_(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrencies', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--anchor', type=int, default=1)
    parser.add_argument('--inject-ms', type=float, nargs='+', default=[0, 10, 20, 40])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(TARGET)
    rows, cfg, _ = original.load_rows('aime25', 16)
    prompts = [tok.encode(original._apply_chat_template(tok, [{'role': 'user', 'content': cfg['question'](r)}], False)) for r in rows]

    data = dict(dataset='aime25', anchor_num=args.anchor, concurrencies=args.concurrencies,
                requests=len(prompts),
                max_new_tokens=args.max_new_tokens, target=TARGET, draft=DRAFT,
                backup='neural', target_gpu=0, draft_gpu=1, cells=[])
    engine = Engine(TARGET, DRAFT)
    ops_per_ms = 1.0/calibrate_inject(1)
    scratch = torch.zeros(*INJECT_SHAPE, device=1)
    data['inject_op'] = f'add_ on {INJECT_SHAPE} cuda:1, {1/ops_per_ms:.3f} ms/op'
    print(f'calibrated inject: {1/ops_per_ms:.3f} ms per dummy add_ op', flush=True)

    predict_spans = []
    recover_spans = []
    rounds = []
    inject_ops = [0]

    original_predict, original_recover = engine._predict, engine._recover
    original_submit, original_receive = engine.target.submit_forward, engine.target.receive

    def timed_predict(seqs):
        torch.cuda.synchronize(1)
        started = time.perf_counter()
        result = original_predict(seqs)
        if inject_ops[0]:
            inject_delay(scratch, inject_ops[0])
        torch.cuda.synchronize(1)
        predict_spans.append((started, time.perf_counter(), len(seqs)))
        return result

    def timed_recover(seqs):
        torch.cuda.synchronize(1)
        started = time.perf_counter()
        original_recover(seqs)
        torch.cuda.synchronize(1)
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

    try:
        for C in args.concurrencies:
            engine.A, engine.micro = args.anchor, C
            print(f'WARMUP C={C} A={args.anchor} micro={C}', flush=True)
            engine.generate(prompts[:C], C, max_new_tokens=32)
            predict_spans.clear(); recover_spans.clear(); rounds.clear()
            engine._predict, engine._recover = timed_predict, timed_recover
            engine.target.submit_forward, engine.target.receive = timed_submit, timed_receive
            engine.target.request('profile_forward', True)
            inject_ops[0] = 0
            print(f'RUN C={C} A={args.anchor} micro={C} verify/draft timing', flush=True)
            result = engine.generate(prompts, C, max_new_tokens=args.max_new_tokens)
            spans = engine.target.request('forward_spans')
            engine.target.request('profile_forward', False)
            engine._predict, engine._recover = original_predict, original_recover
            engine.target.submit_forward, engine.target.receive = original_submit, original_receive

            verify = [1000*(e-s) for s, e in spans]
            draft = [1000*(e-s) for s, e, _ in predict_spans]
            paired = list(zip(spans, predict_spans, rounds))
            exposed = [max(0.0, 1000*(d[1]-t[1])) for t, d, _ in paired]
            hidden = [1.0 if d[1] <= t[1] else 0.0 for t, d, _ in paired]
            round_ms = [1000*(r[1]-r[0]) for r in rounds if r[1]]
            cell = dict(kind='timing', anchor_num=args.anchor, concurrency=C, draft_microbatch=C,
                        tps=result['tps'], rounds=len(spans), verify_forwards=len(spans),
                        verify_ms=stats(verify), draft_ms=stats(draft),
                        round_ms=stats(round_ms), exposed_ms=stats(exposed),
                        hidden_fraction=statistics.fmean(hidden) if hidden else None,
                        slacked_rounds=sum(1 for x in exposed if x > 1e-6),
                        recover_forwards=len(recover_spans),
                        recover_ms=_sum_ms(recover_spans),
                        hit_rate=sum(o['hits'] for o in result['outputs']) /
                                  max(1, sum(o['hits'] for o in result['outputs'])+result['miss_requests']),
                        elapsed_s=result['elapsed_s'])
            data['cells'].append(cell)
            args.output.write_text(json.dumps(data, indent=2))
            print(f"TIMING C={C} A={args.anchor} verify_med={cell['verify_ms']['median']:.2f}ms "
                  f"draft_med={cell['draft_ms']['median']:.2f}ms hidden={cell['hidden_fraction']:.1%} "
                  f"tps={cell['tps']:.1f}", flush=True)

        for C in args.concurrencies:
            engine.A, engine.micro = args.anchor, C
            for ms in args.inject_ms:
                print(f'WARMUP C={C} inject={ms}ms', flush=True)
                inject_ops[0] = 0
                engine.generate(prompts[:C], C, max_new_tokens=32)
                predict_spans.clear(); recover_spans.clear(); rounds.clear()
                engine._predict, engine._recover = timed_predict, timed_recover
                engine.target.submit_forward, engine.target.receive = timed_submit, timed_receive
                inject_ops[0] = int(round(ms*ops_per_ms))
                print(f'RUN C={C} inject={ms}ms', flush=True)
                result = engine.generate(prompts, C, max_new_tokens=args.max_new_tokens)
                engine._predict, engine._recover = original_predict, original_recover
                engine.target.submit_forward, engine.target.receive = original_submit, original_receive
                inject_ops[0] = 0
                cell = dict(kind='inject', anchor_num=args.anchor, concurrency=C, draft_microbatch=C,
                            inject_ms=ms, tps=result['tps'], elapsed_s=result['elapsed_s'],
                            draft_ms=stats([1000*(e-s) for s, e, _ in predict_spans]))
                data['cells'].append(cell)
                args.output.write_text(json.dumps(data, indent=2))
                print(f"INJECT C={C} {ms}ms tps={cell['tps']:.1f}", flush=True)
    finally:
        engine.close()
        args.output.with_suffix('.md').write_text(render(data))
        print(render(data), flush=True)


def _sum_ms(spans):
    total = sum(e-s for s, e, _ in spans)
    return dict(n=len(spans), total_ms=1000*total)


def render(data):
    lines = [f"# micro=C smoke (anchor={data['anchor_num']}, micro batch = concurrency, "
             f"{data['max_new_tokens']} new tokens, {data['requests']} requests)", '',
             '| Concurrency | Verify median ms | Verify p90 | Draft median ms | Draft p90 | Round median ms | '
             'Exposed draft mean ms | Hidden rounds | Recovery forwards | TPS |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in data['cells']:
        if c['kind'] != 'timing':
            continue
        lines.append(f"| {c['concurrency']} | {c['verify_ms']['median']:.2f} | {c['verify_ms']['p90']:.2f} | "
                     f"{c['draft_ms']['median']:.2f} | {c['draft_ms']['p90']:.2f} | {c['round_ms']['median']:.2f} | "
                     f"{c['exposed_ms']['mean']:.2f} | {c['hidden_fraction']:.1%} | {c['recover_forwards']} | {c['tps']:.2f} |")
    lines += ['', '## Injected draft-stream delay vs TPS', '',
              '| Injected ms | C=4 | C=8 | C=16 |', '|---:|---:|---:|---:|']
    concurrencies = sorted({c['concurrency'] for c in data['cells'] if c['kind'] == 'inject'})
    injected = sorted({c['inject_ms'] for c in data['cells'] if c['kind'] == 'inject'})
    lookup = {(c['inject_ms'], c['concurrency']): c['tps'] for c in data['cells'] if c['kind'] == 'inject'}
    for ms in injected:
        cells = ' | '.join(f"{lookup.get((ms, c), float('nan')):.1f}" for c in concurrencies)
        lines.append(f"| {ms:.0f} | {cells} |")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    main()
