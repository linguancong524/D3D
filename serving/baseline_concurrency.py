#!/usr/bin/env python3
"""End-to-end AR baseline with bounded admission, natural EOS and refill."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer, GenerationConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'single'))
sys.path.insert(0, str(ROOT))
from serving.backend import Sequence, SamplingParams
from serving.target_process import TargetProcess
from single import eval_sync as original


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def generate(worker, prompts, concurrency, limit, eos, progress=None, ignore_eos=False):
    if concurrency < 1 or limit < 1 or not prompts:
        raise ValueError('Nonempty prompts and positive concurrency/token limit required')
    started = time.perf_counter()
    worker.request('reset_stats')
    waiting = iter(enumerate(prompts))
    running, outputs, batches = [], [], []
    exhausted = False
    max_active = 0
    last_progress = started

    def emit(seq, token):
        seq.generated.append(token)
        hit_eos = (not ignore_eos) and token in eos
        if hit_eos or len(seq.generated) >= limit:
            outputs.append(dict(index=seq.index, tokens=seq.generated,
                                latency_s=time.perf_counter() - seq.born,
                                finish_reason='eos' if hit_eos else 'length'))
            worker.release(seq)
            return True
        return False

    while running or not exhausted:
        admitted = []
        while len(running) + len(admitted) < concurrency and not exhausted:
            try:
                index, ids = next(waiting)
            except StopIteration:
                exhausted = True
                break
            seq = Sequence(ids, SamplingParams(temperature=0, max_new_tokens=limit))
            seq.index, seq.prefix = index, len(ids)
            seq.generated, seq.born = [], time.perf_counter()
            admitted.append(seq)
        max_active = max(max_active, len(running) + len(admitted))
        # Match the speculative engine's admission prefill microbatch size.
        for offset in range(0, len(admitted), 4):
            group = admitted[offset:offset + 4]
            for seq, token in zip(group, worker.prefill(group)):
                if not emit(seq, token):
                    running.append(seq)
        if running:
            batches.append(len(running))
            worker.submit_forward(running, [[seq.generated[-1]] for seq in running])
            next_tokens = worker.receive()
            remaining = []
            for seq, tokens in zip(running, next_tokens):
                seq.prefix += 1
                if not emit(seq, tokens[0]):
                    remaining.append(seq)
            running = remaining
        if progress and time.perf_counter() - last_progress >= 60:
            progress(f'PROGRESS completed={len(outputs)}/{len(prompts)} active={len(running)} '
                     f'elapsed={time.perf_counter() - started:.1f}s')
            last_progress = time.perf_counter()
    stats = worker.request('stats')
    elapsed = time.perf_counter() - started
    if stats['active_sequences']:
        raise RuntimeError('Target KV sequences leaked')
    outputs.sort(key=lambda item: item['index'])
    count = sum(len(item['tokens']) for item in outputs)
    latency = sorted(item['latency_s'] for item in outputs)
    return dict(outputs=outputs, elapsed_s=elapsed, output_tokens=count, tps=count / elapsed,
                requests=len(prompts), max_active=max_active, decode_batch_sizes=batches,
                latency_p50_s=statistics.median(latency),
                latency_p95_s=latency[min(len(latency) - 1, int(0.95 * len(latency)))],
                length_limit_fraction=sum(item['finish_reason'] == 'length' for item in outputs) / len(outputs),
                mean_output_tokens=count / len(outputs), **stats)


def slow_attempts(cells, fraction):
    reference = (max(cell['tps'] for cell in cells) if len(cells) == 2
                 else statistics.median(cell['tps'] for cell in cells))
    return [cell['attempt'] for cell in cells if cell['tps'] < fraction * reference]


def telemetry():
    return subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used',
                           '--format=csv'], capture_output=True, text=True, check=False).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default=os.environ.get('D3D_TARGET_MODEL', 'Qwen/Qwen3-8B'))
    parser.add_argument('--datasets', nargs='+', default=['math500', 'humaneval', 'mt-bench'])
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 4, 8, 16])
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--warmup-new-tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-retests', type=int, default=1)
    parser.add_argument('--slow-fraction', type=float, default=0.85)
    parser.add_argument('--pages', type=int, default=512)
    parser.add_argument('--reference-device', default='cuda:0',
                        help='device for the independent HF validation model; use another GPU when the '
                             'verifier itself nearly fills one card (Qwen3-32B)')
    parser.add_argument('--skip-reference-validation', action='store_true',
                        help='skip the independent HF greedy check when no second card is available; '
                             'the run then reports validation as skipped')
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--output-root', type=Path, default=HERE / 'results_baseline')
    args = parser.parse_args()
    if min(args.concurrency + [args.repeats, args.max_new_tokens, args.warmup_new_tokens, args.pages]) < 1 or args.max_samples < 0 or args.max_retests < 0:
        parser.error('Invalid numeric parameter')
    if not 0 < args.slow_fraction < 1:
        parser.error('slow-fraction must be between zero and one')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    config = GenerationConfig.from_pretrained(args.target)
    eos = config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    datasets = {}
    directories = {}
    for name in args.datasets:
        rows, cfg, source = original.load_rows(name, args.max_samples or None)
        prompts = [tokenizer.encode(original._apply_chat_template(
            tokenizer, [{'role': 'user', 'content': cfg['question'](row)}], False)) for row in rows]
        if not prompts:
            raise ValueError(f'No prompts: {name}')
        worst_pages = sum(sorted(((len(ids) + args.max_new_tokens + 255) // 256 for ids in prompts), reverse=True)[:max(args.concurrency)])
        if worst_pages > args.pages or max(map(len, prompts)) + args.max_new_tokens > 16384:
            raise ValueError(f'{name}: KV capacity/context limit insufficient; worst_pages={worst_pages}')
        directory = args.output_root / name / args.run_id
        directory.mkdir(parents=True, exist_ok=False)
        directories[name] = directory
        datasets[name] = prompts
        save(directory / 'inputs.json', dict(source=str(source), prompts=prompts))
        print(f'DATASET {name}: {len(prompts)} prompts, max_input={max(map(len, prompts))}, worst_pages={worst_pages}', flush=True)
    metadata = dict(config={**vars(args), 'output_root': str(args.output_root)},
                    engine='SSD-derived custom PyTorch; persistent target process; eager; no CUDA graphs',
                    metric='actual output tokens / wall seconds including prefill, decode, drain and cleanup; excludes loading, tokenization, warmup, validation and serialization',
                    hardware=torch.cuda.get_device_name(0), cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                    torch=torch.__version__, temperature=0, enable_thinking=False, eos_ids=sorted(eos),
                    latency_definition='admission to completion; excludes waiting before admission',
                    repeat_policy='All attempts retained; median TPS and pooled TPS include original and retest attempts. No fastest-run selection.',
                    slowdown_policy='For two initial attempts, slower TPS < 0.85 * faster TPS triggers one additional attempt in the suite; final flags use median for three attempts.',
                    source_sha256={str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in HERE.rglob('*.py')}, cells=[])
    worker = TargetProcess(args.target, args.pages)
    try:
        worker.receive()
        # Independent short greedy references establish token alignment before timing.
        # The reference is a second full copy of the verifier, so on a card that the
        # verifier already nearly fills it has to live on another device or be skipped.
        checks = []
        if args.skip_reference_validation:
            print('VALIDATION SKIPPED: no independent HF reference (requested)', flush=True)
        else:
            from transformers import AutoModelForCausalLM
            reference = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=torch.bfloat16,
                                                           attn_implementation='sdpa').to(args.reference_device).eval()
            for name, prompts in datasets.items():
                cases = prompts[:2]
                actual = generate(worker, cases, 2, 16, eos)
                for ids, output in zip(cases, actual['outputs']):
                    x = torch.tensor([ids], device=args.reference_device)
                    with torch.inference_mode():
                        expected = reference.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=16, do_sample=False,
                                                      eos_token_id=sorted(eos), pad_token_id=tokenizer.pad_token_id)[0, len(ids):].tolist()
                    check = dict(dataset=name, index=output['index'], equal=output['tokens'] == expected,
                                 actual=output['tokens'], expected=expected)
                    if not check['equal']:
                        # Score the actual path to distinguish near-tie BF16 differences
                        # from token-position/cache errors without changing the workload.
                        actual_tokens = output['tokens']
                        teacher = torch.tensor([ids + actual_tokens[:-1]], device=args.reference_device)
                        with torch.inference_mode():
                            logits = reference(teacher).logits[0, len(ids) - 1:].float()
                        chosen = torch.tensor(actual_tokens, device=logits.device)
                        regret = logits.max(-1).values - logits.gather(1, chosen[:, None]).squeeze(1)
                        check['teacher_forced_logit_regret'] = regret.tolist()
                        check['max_logit_regret'] = regret.max().item()
                        check['near_tie_tolerance'] = 0.0
                        check['numerically_consistent'] = check['max_logit_regret'] == 0.0
                        print(f'VALIDATION DIFFERENCE {name}: max_logit_regret={check["max_logit_regret"]}', flush=True)
                    checks.append(check)
            del reference
            torch.cuda.empty_cache()
        for directory in directories.values():
            save(directory / 'validation.json', checks)
        if checks and not all(check['equal'] or check.get('numerically_consistent', False) for check in checks):
            raise RuntimeError('Independent HF greedy validation failed')
        if checks:
            print(f'VALIDATION PASSED: {sum(check["equal"] for check in checks)}/{len(checks)} exact; '
                  'remaining paths checked against independent teacher-forced argmax ties', flush=True)
        for name, prompts in datasets.items():
            directory = directories[name]
            payload = {**metadata, 'dataset': name, 'requests': len(prompts), 'cells': [], 'status': 'running',
                       'input_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest()}

            def log(message):
                line = f'{datetime.now(timezone.utc).isoformat()} {message}'
                print(line, flush=True)
                with (directory / 'run.log').open('a') as stream:
                    stream.write(line + '\n')

            save(directory / 'benchmark.json', payload)
            for concurrency in args.concurrency:
                log(f'WARMUP {name} C={concurrency}')
                # Warm the same full request workload at this concurrency. The
                # result is discarded and its time never enters a cell.
                generate(worker, prompts, concurrency, args.warmup_new_tokens, eos)
                cells = []
                retest_target = 0
                for attempt in range(args.repeats + args.max_retests):
                    if attempt >= args.repeats:
                        if attempt == args.repeats:
                            retest_target = args.max_retests if slow_attempts(cells, args.slow_fraction) else 0
                        if attempt >= args.repeats + retest_target:
                            break
                        log(f'RETEST low_attempts={slow_attempts(cells, args.slow_fraction)} threshold={args.slow_fraction}')
                        generate(worker, prompts, concurrency, args.warmup_new_tokens, eos)
                    log(f'RUN {name} C={concurrency} attempt={attempt}')
                    before = telemetry()
                    result = generate(worker, prompts, concurrency, args.max_new_tokens, eos, log)
                    result.update(dataset=name, concurrency=concurrency, attempt=attempt,
                                  is_retest=attempt >= args.repeats, telemetry_before=before, telemetry_after=telemetry())
                    if result['free_pages'] != args.pages or result['max_active'] > concurrency:
                        raise RuntimeError('KV release or admission invariant failed')
                    raw_path = directory / f'c{concurrency}_attempt{attempt}.json'
                    save(raw_path, result)
                    cell = {key: value for key, value in result.items()
                            if key not in ('outputs', 'decode_batch_sizes', 'telemetry_before', 'telemetry_after')}
                    cell['raw_file'] = raw_path.name
                    cells.append(cell)
                    payload['cells'].append(cell)
                    save(directory / 'benchmark.json', payload)
                    log(f'RESULT {name} C={concurrency} attempt={attempt} TPS={result["tps"]:.2f} elapsed={result["elapsed_s"]:.1f}s')
                payload.setdefault('summary', {})[str(concurrency)] = dict(
                    median_tps=statistics.median(cell['tps'] for cell in cells),
                    pooled_tps=sum(cell['output_tokens'] for cell in cells) / sum(cell['elapsed_s'] for cell in cells),
                    min_tps=min(cell['tps'] for cell in cells), max_tps=max(cell['tps'] for cell in cells),
                    attempts=len(cells), slow_attempts=slow_attempts(cells, args.slow_fraction))
                save(directory / 'benchmark.json', payload)
                lines = ['| Concurrency | Median TPS | Pooled TPS | Min TPS | Max TPS | Attempts | Slow attempts |',
                         '|---:|---:|---:|---:|---:|---:|---|']
                for key, summary in payload['summary'].items():
                    lines.append(f'| {key} | {summary["median_tps"]:.2f} | {summary["pooled_tps"]:.2f} | '
                                 f'{summary["min_tps"]:.2f} | {summary["max_tps"]:.2f} | '
                                 f'{summary["attempts"]} | {summary["slow_attempts"]} |')
                (directory / 'table.md').write_text('\n'.join(lines) + '\n')
            payload['status'] = 'complete'
            save(directory / 'benchmark.json', payload)
        print('ALL DATASETS COMPLETE', flush=True)
    except BaseException as error:
        for directory in directories.values():
            path = directory / 'benchmark.json'
            state = json.loads(path.read_text()) if path.exists() else {**metadata, 'cells': []}
            if state.get('status') != 'complete':
                state.update(status='failed', error=repr(error))
                save(path, state)
        raise
    finally:
        worker.close()


if __name__ == '__main__':
    main()
