#!/usr/bin/env python3
"""Full-sample speculative concurrency suite for the 1.7B draft pairs.

Protocol matches results_baseline (scripts/baseline_concurrency.sh): discarded
256-token full-workload warmup per cell, `--repeats` attempts, one automatic
retest when the slower attempt is below --slow-fraction of the faster one, and
cells reported as pooled output tokens / pooled wall time over all attempts.

The draft microbatch is the full concurrency (the 2026-09-21 ablation showed
micro=C is never worse), the anchor is per concurrency (`--anchor-map`), and the
32B pair needs `--pages 96`: 62 GB of bf16 verifier weights plus a 512-page KV
pool would not fit one 80 GB card.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from bench import original, AutoTokenizer, Engine

HERE = Path(__file__).resolve().parent


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def telemetry():
    return subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used',
                           '--format=csv'], capture_output=True, text=True, check=False).stdout


def parse_anchor_map(text):
    return {int(item.split(':')[0]): int(item.split(':')[1]) for item in text.split(',')}


def compact_cell(result, **extra):
    """Timing/summary view of one attempt, without per-request token lists."""
    latencies = sorted(o['latency_s'] for o in result['outputs'])
    accepts = [n for o in result['outputs'] for n in o['accepts']]
    hits = sum(o['hits'] for o in result['outputs'])
    return dict(**extra, elapsed_s=result['elapsed_s'], output_tokens=result['output_tokens'],
                requests=len(result['outputs']), tps=result['tps'], max_active=result['max_active'],
                decode_batch_sizes=result['decode_batch_sizes'],
                mean_output_tokens=result['output_tokens']/len(result['outputs']),
                latency_p50_s=latencies[len(latencies)//2], latency_p95_s=latencies[int(0.95*(len(latencies)-1))],
                hit_rate=hits/(hits+result['miss_requests']) if hits+result['miss_requests'] else 0,
                miss_rounds=result['miss_rounds'], miss_requests=result['miss_requests'],
                recoveries=sum(o['recoveries'] for o in result['outputs']),
                mean_accept=statistics.fmean(accepts) if accepts else 0,
                peak_allocated_gb=result['peak_allocated_gb'],
                target_free_pages=result['target_free_pages'],
                prefill_s=result.get('prefill_s'), decode_s=result.get('decode_s'),
                decode_tps=result.get('decode_tps'), admission_groups=result.get('admission_groups'),
                finish_reason={o['index']: ('eos' if o['tokens'] and o['tokens'][-1] in (151643, 151645) else 'limit')
                               for o in result['outputs']})


class PhaseTimer:
    """Split wall time into admission/prefill vs decode without touching the engine.

    The engine prefills each admission group synchronously inside the main loop,
    so wrapping submit_prefill -> receive measures the whole blocking block, both
    models included. Everything else in generate() is decode, drain and cleanup.
    """

    def __init__(self, engine):
        self.engine = engine
        self.prefill_s = 0.0
        self.groups = 0
        self._started = None
        self._submit = engine.target.submit_prefill
        self._receive = engine.target.receive

    def __enter__(self):
        def submit_prefill(seqs):
            self._started = time.perf_counter()
            self.groups += 1
            return self._submit(seqs)

        def receive():
            result = self._receive()
            if self._started is not None:
                self.prefill_s += time.perf_counter() - self._started
                self._started = None
            return result

        self.engine.target.submit_prefill = submit_prefill
        self.engine.target.receive = receive
        return self

    def __exit__(self, *exc):
        self.engine.target.submit_prefill = self._submit
        self.engine.target.receive = self._receive
        return False


def full_cell(result, **extra):
    """Audit view: keeps per-request tokens and the fields audit_results.py wants."""
    latencies = sorted(o['latency_s'] for o in result['outputs'])
    return dict(**extra, elapsed_s=result['elapsed_s'], output_tokens=result['output_tokens'],
                requests=len(result['outputs']), tps=result['tps'], max_active=result['max_active'],
                decode_batch_sizes=result['decode_batch_sizes'],
                mean_output_tokens=result['output_tokens']/len(result['outputs']),
                latency_p50_s=latencies[len(latencies)//2], latency_p95_s=latencies[int(0.95*(len(latencies)-1))],
                peak_allocated_gb=result['peak_allocated_gb'],
                outputs=[dict(index=o['index'], tokens=o['tokens'], latency_s=o['latency_s'],
                              hits=o['hits'], recoveries=o['recoveries']) for o in result['outputs']])


def render_table(payload):
    cfg = payload['config']
    concurrencies = sorted({c['concurrency'] for c in payload['cells']})
    lines = [f"# {payload['pair']} speculative suite, {payload['dataset']} ({payload['requests']} prompts)",
             '',
             f"max_new_tokens={cfg['max_new_tokens']}, warmup={cfg['warmup_new_tokens']} discarded, "
             f"repeats={cfg['repeats']} + up to {cfg['max_retests']} retest (slower < "
             f"{cfg['slow_fraction']:.2f} x faster), draft microbatch = concurrency, backup=neural.",
             '',
             '| Concurrency | Anchor | Attempts | TPS per attempt | Pooled TPS | Decode-only TPS | Hit rate | Mean accept | Prefill s | Wall s |',
             '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in concurrencies:
        cells = [cell for cell in payload['cells'] if cell['concurrency'] == c]
        tokens = sum(cell['output_tokens'] for cell in cells)
        wall = sum(cell['elapsed_s'] for cell in cells)
        pooled = tokens/wall
        prefill = sum(cell['prefill_s'] or 0 for cell in cells)
        decode = sum((cell['decode_s'] or 0) for cell in cells)
        decode_tps = f"{tokens/decode:.1f}" if decode > 0 else 'n/a'
        lines.append(f"| {c} | {cells[0]['anchor_num']} | {len(cells)} | "
                     f"{', '.join(f'{cell['tps']:.1f}' for cell in cells)} | {pooled:.2f} | "
                     f"{decode_tps} | {cells[0]['hit_rate']:.1%} | {cells[0]['mean_accept']:.2f} | "
                     f"{prefill:.1f} | {wall:.1f} |")
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--pair', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--pages', type=int, default=512)
    parser.add_argument('--datasets', nargs='+', default=['math500', 'humaneval', 'mt-bench'])
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--anchor-map', default='1:1,4:1,8:1,16:1')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--warmup-new-tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-retests', type=int, default=1)
    parser.add_argument('--slow-fraction', type=float, default=0.85)
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    anchor_map = parse_anchor_map(args.anchor_map)
    missing = [c for c in args.concurrency if c not in anchor_map]
    if missing:
        parser.error(f'anchor-map missing concurrency {missing}')

    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    datasets, directories = {}, {}
    for name in args.datasets:
        rows, cfg, source = original.load_rows(name, args.max_samples or None)
        prompts = [tokenizer.encode(original._apply_chat_template(
            tokenizer, [{'role': 'user', 'content': cfg['question'](row)}], False)) for row in rows]
        directory = args.output_root / name / args.run_id
        directory.mkdir(parents=True, exist_ok=True)
        directories[name] = directory
        datasets[name] = prompts
        save(directory / 'inputs.json', dict(source=str(source), prompts=prompts))
        worst = sum(sorted(((len(ids)+args.max_new_tokens+255)//256 for ids in prompts), reverse=True)[:max(args.concurrency)])
        print(f'DATASET {name}: {len(prompts)} prompts, max_input={max(map(len, prompts))}, '
              f'worst_pages={worst}, pages={args.pages}', flush=True)
        if worst > args.pages:
            raise ValueError(f'{name}: KV capacity insufficient; worst_pages={worst}')

    metadata = dict(config={**vars(args), 'output_root': str(args.output_root), 'anchor_map': anchor_map},
                    pair=args.pair, engine='D3D strict-concurrency speculative engine; persistent target process on '
                                           'cuda:0, SDAR draft on cuda:1; eager, no CUDA graphs',
                    metric='actual output tokens / wall seconds including prefill, decode and bootstrap; '
                           'excludes model loading, tokenization and warmup',
                    hardware=[torch.cuda.get_device_name(0), torch.cuda.get_device_name(1)],
                    cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                    torch=torch.__version__, temperature=0, enable_thinking=False, backup='neural',
                    latency_definition='admission to completion; excludes waiting before admission',
                    repeat_policy='All attempts retained; pooled TPS is total output tokens over total wall time. '
                                  'No fastest-run selection.',
                    slowdown_policy='Slower TPS < fraction x faster TPS after the initial attempts triggers one '
                                    'additional attempt per cell.',
                    source_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [HERE/'sweep_full.py', HERE/'engine.py', HERE/'backend.py', HERE/'target_process.py']},
                    cells=[])
    audit = {'config': dict(datasets=args.datasets, concurrency=args.concurrency, repeats=args.repeats,
                            max_new_tokens=args.max_new_tokens, ignore_eos=False), 'cells': []}

    engine = Engine(args.target, args.draft, pages=args.pages)
    try:
        for name, prompts in datasets.items():
            directory = directories[name]
            payload = {**metadata, 'dataset': name, 'requests': len(prompts), 'cells': [], 'status': 'running',
                       'input_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest()}

            def log(message):
                line = f'{datetime.now(timezone.utc).isoformat()} {message}'
                print(line, flush=True)
                with (directory / 'run.log').open('a') as stream:
                    stream.write(line + '\n')

            benchmark = directory / 'benchmark.json'
            # A cell is finished only once its own marker exists; attempt-level
            # entries in benchmark.json are not enough to resume safely.
            finished = ({int(path.stem[1:].split('_')[0]) for path in directory.glob('c*_complete.json')}
                        if args.skip_existing else set())
            if args.skip_existing and benchmark.exists():
                previous = json.loads(benchmark.read_text())
                if previous.get('status') == 'complete':
                    log(f'SKIP {name}: complete')
                    continue
                payload['cells'] = [cell for cell in previous.get('cells', [])
                                    if cell['concurrency'] in finished]
                audit_path = args.output_root / f'audit_{args.pair}.json'
                if audit_path.exists():
                    stored = json.loads(audit_path.read_text())
                    stored['cells'] = [cell for cell in stored.get('cells', [])
                                       if cell['dataset'] != name or cell['concurrency'] in finished]
                    audit = stored
                if finished:
                    log(f'RESUME {name}: keeping finished cells C={sorted(finished)}')
            save(benchmark, payload)
            log(f'START {name}: {len(prompts)} prompts, concurrency {args.concurrency}')

            for concurrency in args.concurrency:
                if concurrency in finished:
                    log(f'SKIP {name} C={concurrency}: already measured in {directory.name}')
                    continue
                anchor = anchor_map[concurrency]
                engine.A, engine.micro = anchor, concurrency
                log(f'WARMUP {name} C={concurrency} anchor={anchor} micro={concurrency}')
                engine.generate(prompts, concurrency, max_new_tokens=args.warmup_new_tokens)
                cells = []
                attempt = 0
                retests = 0
                while attempt < max(args.repeats, 1) + retests:
                    is_retest = attempt >= args.repeats
                    before = telemetry()
                    log(f'RUN {name} C={concurrency} anchor={anchor} attempt={attempt}'
                        + (' (retest)' if is_retest else ''))
                    started = datetime.now(timezone.utc).isoformat()
                    with PhaseTimer(engine) as timer:
                        result = engine.generate(prompts, concurrency, max_new_tokens=args.max_new_tokens)
                    result['prefill_s'] = timer.prefill_s
                    result['decodephase_s'] = result['elapsed_s'] - timer.prefill_s
                    result['decode_s'] = result['decodephase_s']
                    result['decode_tps'] = result['output_tokens'] / result['decodephase_s'] if result['decodephase_s'] > 0 else 0
                    result['admission_groups'] = timer.groups
                    extra = dict(dataset=name, concurrency=concurrency, repeat=attempt, attempt=attempt,
                                 is_retest=is_retest, anchor_num=anchor, draft_microbatch=concurrency,
                                 max_new_tokens=args.max_new_tokens, ignore_eos=False,
                                 input_sha256=payload['input_sha256'], started_utc=started,
                                 telemetry_before=before, telemetry_after=telemetry())
                    save(directory / f'c{concurrency}_attempt{attempt}.json', full_cell(result, **extra))
                    cell = compact_cell(result, **extra)
                    cells.append(cell)
                    payload['cells'].append(cell)
                    audit['cells'].append(full_cell(result, **extra))
                    save(benchmark, payload)
                    (directory / 'table.md').write_text(render_table(payload))
                    save(args.output_root / f'audit_{args.pair}.json', audit)
                    log(f'RESULT {name} C={concurrency} attempt={attempt} tps={result["tps"]:.2f} '
                        f'hit={cell["hit_rate"]:.3f} accept={cell["mean_accept"]:.2f} '
                        f'tokens={result["output_tokens"]} elapsed={result["elapsed_s"]:.1f}s')
                    attempt += 1
                    if attempt >= args.repeats and retests < args.max_retests and len(cells) >= 2:
                        worst, best = min(c['tps'] for c in cells), max(c['tps'] for c in cells)
                        if worst < args.slow_fraction*best:
                            retests += 1
                            log(f'RETEST {name} C={concurrency}: slowest {worst:.2f} < '
                                f'{args.slow_fraction:.2f} x {best:.2f}')
                pooled_tokens = sum(c['output_tokens'] for c in cells)
                pooled_wall = sum(c['elapsed_s'] for c in cells)
                save(directory / f'c{concurrency}_complete.json',
                     dict(concurrency=concurrency, anchor=anchor, attempts=len(cells),
                          pooled_tps=pooled_tokens/pooled_wall))
                log(f'CELL {name} C={concurrency} pooled_tps={pooled_tokens/pooled_wall:.2f} attempts={len(cells)}')
            payload['status'] = 'complete'
            save(benchmark, payload)
            (directory / 'table.md').write_text(render_table(payload))
            log(f'DONE {name}')
    finally:
        engine.close()
        save(args.output_root / f'audit_{args.pair}.json', audit)


if __name__ == '__main__':
    main()
