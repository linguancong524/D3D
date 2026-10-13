#!/usr/bin/env python3
"""Fixed-batch decode-only AR throughput, using the production target worker."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from serving.target_process import TargetProcess
from serving.backend import Sequence, SamplingParams
from common.data import load_rows
from common.text import apply_chat_template


def decode(worker, prompts, steps, capture=False):
    seqs = [Sequence(ids, SamplingParams(temperature=0)) for ids in prompts]
    for s in seqs:
        s.prefix = len(s.token_ids)
    anchors = worker.prefill(seqs)
    blocks = [[a] for a in anchors]
    tokens = [[a] for a in anchors] if capture else None
    # Prefill's first output is excluded from both decode time and token count.
    started = time.perf_counter()
    for _ in range(steps):
        worker.submit_forward(seqs, blocks)
        blocks = worker.receive()
        for i, s in enumerate(seqs):
            s.prefix += 1
            if capture:
                tokens[i].append(blocks[i][0])
    elapsed = time.perf_counter() - started
    for s in seqs:
        worker.release(s)
    stats = worker.request('stats')
    if stats['active_sequences']:
        raise RuntimeError('Target KV sequences leaked')
    return {'elapsed_s': elapsed, 'tokens': len(prompts) * steps,
            'requests': len(prompts), 'peak_allocated_gb': stats['peak_allocated_gb'],
            'free_pages': stats['free_pages'], 'outputs': tokens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', default=os.environ.get('TARGET_MODEL', 'Qwen/Qwen3-8B'))
    p.add_argument('--datasets', nargs='+', default=['aime25'])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    p.add_argument('--max-samples', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--pages', type=int, default=512)
    p.add_argument('--output', default=str(HERE / 'results_baseline/qwen3_8b_decode.json'))
    a = p.parse_args()
    if min(a.batch_sizes + [a.max_samples, a.repeats, a.pages]) < 1 or a.max_new_tokens < 2:
        p.error('All numeric parameters must be positive')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.target)
    datasets = {}
    sampling = {}
    for name in a.datasets:
        rows, cfg, _ = load_rows(name, a.max_samples)
        texts = [apply_chat_template(tok, [{'role': 'user', 'content': cfg['question'](r)}], False)
                 for r in rows]
        if not texts:
            raise ValueError(f'No prompts for {name}')
        indices = [i % len(texts) for i in range(a.max_samples)]
        datasets[name] = [tok.encode(texts[i]) for i in indices]
        sampling[name] = {'unique_source_rows': len(rows), 'source_indices': indices}
        if any(a.max_samples % b for b in a.batch_sizes):
            p.error(f'{name}: sample count must be divisible by each batch size')
    payload = {'config': vars(a), 'engine': 'SSD-derived custom PyTorch; persistent target process',
               'hardware': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'metric': 'decode output tokens / decode wall seconds; includes per-step IPC and CPU scheduling; excludes prefill, first output token, cleanup, loading and warmup',
               'ignore_eos': True, 'temperature': 0, 'enable_thinking': False, 'sampling': sampling,
               'source_sha256': {str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in HERE.rglob('*.py')}, 'cells': []}
    worker = TargetProcess(a.target, a.pages)
    try:
        worker.receive()
        # Independent greedy reference check outside throughput measurement.
        from transformers import AutoModelForCausalLM
        check = next(iter(datasets.values()))[:2]
        actual = decode(worker, check, 8, capture=True)['outputs']
        ref = AutoModelForCausalLM.from_pretrained(a.target, torch_dtype=torch.bfloat16,
                                                  attn_implementation='sdpa').to('cuda:0').eval()
        checks = []
        for ids, got in zip(check, actual):
            with torch.inference_mode():
                x = torch.tensor([ids], device='cuda:0')
                expected = ref.generate(x, max_new_tokens=9, do_sample=False,
                                        eos_token_id=None, pad_token_id=151643)[0, len(ids):].tolist()
            checks.append({'equal': got == expected, 'actual': got, 'expected': expected})
        del ref
        torch.cuda.empty_cache()
        payload['validation'] = checks
        out.write_text(json.dumps(payload, indent=2))
        if not all(c['equal'] for c in checks):
            raise RuntimeError('Independent HF greedy validation failed')
        print('VALIDATION PASSED', flush=True)
        for name, prompts in datasets.items():
            for b in a.batch_sizes:
                decode(worker, prompts[:b], 32)
                for repeat in range(a.repeats):
                    worker.request('reset_stats')
                    groups = [decode(worker, prompts[i:i+b], a.max_new_tokens - 1)
                              for i in range(0, len(prompts), b)]
                    count = sum(g['tokens'] for g in groups)
                    elapsed = sum(g['elapsed_s'] for g in groups)
                    cell = {'dataset': name, 'batch_size': b, 'repeat': repeat,
                            'tokens': count, 'elapsed_s': elapsed, 'tps': count / elapsed,
                            'input_sha256': hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                            'groups': groups}
                    payload['cells'].append(cell)
                    out.write_text(json.dumps(payload, indent=2))
                    print(f'RESULT {name} B={b} repeat={repeat}: {cell["tps"]:.2f} tok/s', flush=True)
        lines = ['# Qwen3-8B AR decoding throughput', '',
                 'SSD-derived custom PyTorch, one H800, BF16 greedy. Fixed batches, EOS ignored.',
                 f'{a.max_samples} prompts per dataset, {a.max_new_tokens} output tokens '
                 f'({a.max_new_tokens - 1} timed decode steps), {a.repeats} repeats. '
                 'Prefill and its first output token excluded. Short datasets cycle from the start; '
                 'source indices are recorded in JSON.', '',
                 '| Batch | ' + ' | '.join(a.datasets) + ' |',
                 '|---|' + '---:|' * len(a.datasets)]
        for b in a.batch_sizes:
            values = []
            for name in a.datasets:
                cells = [c for c in payload['cells'] if c['batch_size'] == b and c['dataset'] == name]
                values.append(f'{sum(c["tokens"] for c in cells) / sum(c["elapsed_s"] for c in cells):.1f}')
            lines.append(f'| {b} | ' + ' | '.join(values) + ' |')
        out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
        print('DONE', out, flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
