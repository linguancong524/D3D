"""Anchor smoke test for the 1.7B drafter -> 32B verifier pair.

Same harness as sweep_anchor.py (full-concurrency draft microbatch, warmup
discarded), but the target does not fit with the default pool: Qwen3-32B is
62 GB of bf16 weights and the KV pool costs pages*256*kv_heads*head_dim*2*2
per layer, so 512 pages would be another 34 GB. Default here is pages=96
(~6.4 GB), which is plenty for low concurrency short runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from bench import original, AutoTokenizer, Engine

TARGET_32B = str(Path('/inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/models/Qwen3-32B'))
DRAFT_1_7B = ('/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/'
              'qwen3-32b/sdar17b_b16_a128_ar1.0_from255044_ce_3epoch_4gpu/step-1041183')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', default=TARGET_32B)
    parser.add_argument('--draft', default=DRAFT_1_7B)
    parser.add_argument('--pages', type=int, default=96)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--anchors', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--dataset', default='aime25')
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--bidirectional', action='store_true',
                        help='repeat the anchor order in reverse to spread host drift')
    parser.add_argument('--pairs', type=int, default=0,
                        help='alternate two anchors in back-to-back pairs (paired comparison under drift)')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(args.target)
    rows, cfg, _ = original.load_rows(args.dataset, args.rows)
    prompts = [tok.encode(original._apply_chat_template(tok, [{'role': 'user', 'content': cfg['question'](r)}], False)) for r in rows]
    data = dict(dataset=args.dataset, requests=len(prompts), max_new_tokens=args.max_new_tokens,
                target=args.target, draft=args.draft, pages=args.pages, concurrency=args.concurrency,
                backup='neural',
                metric='output tokens / wall seconds including prefill and bootstrap; excludes warmup and model loading',
                input_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                hardware=[torch.cuda.get_device_name(i) for i in (0, 1)], cells=[])
    if args.pairs:
        if len(args.anchors) != 2:
            parser.error('--pairs needs exactly two anchors')
        a, b = args.anchors
        order = [x for i in range(args.pairs) for x in ((a, b) if i % 2 == 0 else (b, a))]
    else:
        order = list(args.anchors) + (list(reversed(args.anchors)) if args.bidirectional else [])
    data['order'] = order

    engine = Engine(args.target, args.draft, pages=args.pages)
    try:
        print(f'engine up: target GPU0 KV pool {engine.target.request("stats")["peak_allocated_gb"]:.2f} GB peak', flush=True)
        for anchor in order:
            engine.A, engine.micro = anchor, args.concurrency
            print(f'WARMUP C={args.concurrency} A={anchor} micro={engine.micro}', flush=True)
            engine.generate(prompts[:args.concurrency], args.concurrency, max_new_tokens=32)
            print(f'RUN C={args.concurrency} A={anchor} micro={engine.micro}', flush=True)
            result = engine.generate(prompts, args.concurrency, max_new_tokens=args.max_new_tokens)
            hits = sum(o['hits'] for o in result['outputs'])
            accepts = [n for o in result['outputs'] for n in o['accepts']]
            recoveries = sum(o['recoveries'] for o in result['outputs'])
            result.update(concurrency=args.concurrency, anchor_num=anchor,
                          draft_microbatch=engine.micro,
                          hit_rate=hits/(hits+result['miss_requests']) if hits+result['miss_requests'] else 0,
                          mean_accept=sum(accepts)/len(accepts) if accepts else 0,
                          recoveries=recoveries)
            data['cells'].append(result)
            args.output.write_text(json.dumps(data, indent=2))
            args.output.with_suffix('.md').write_text(render(data))
            print(f"RESULT C={args.concurrency} A={anchor} TPS={result['tps']:.2f} hit={result['hit_rate']:.3f} "
                  f"accept={result['mean_accept']:.2f} recover={recoveries} peak={result['peak_allocated_gb']}", flush=True)
    finally:
        engine.close()
        print(render(data), flush=True)


def render(data):
    lines = [f"# 1.7B -> 32B anchor smoke ({data['dataset']}, {data['requests']} requests, "
             f"C={data['concurrency']}, {data['max_new_tokens']} new tokens, pages={data['pages']})", '',
             '| Anchor | Microbatch | TPS | Hit rate | Mean accept | Recoveries | Target peak GB | Draft peak GB |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in data['cells']:
        lines.append(f"| {c['anchor_num']} | {c['draft_microbatch']} | {c['tps']:.2f} | {c['hit_rate']:.1%} | "
                     f"{c['mean_accept']:.2f} | {c['recoveries']} | {c['peak_allocated_gb'][0]:.2f} | {c['peak_allocated_gb'][1]:.2f} |")
    anchors = sorted({c['anchor_num'] for c in data['cells']})
    if len(anchors) == 2 and len(data['cells']) % 2 == 0:
        a, b = anchors
        lines += ['', f'## Paired comparison (back-to-back, {len(data["cells"])//2} pairs)', '',
                  '| Pair | A%d TPS | A%d TPS | A%d advantage |' % (a, b, a), '|---:|---:|---:|---:|']
        deltas = []
        for i in range(0, len(data['cells']), 2):
            first, second = data['cells'][i], data['cells'][i+1]
            ta = first['tps'] if first['anchor_num'] == a else second['tps']
            tb = second['tps'] if first['anchor_num'] == a else first['tps']
            advantage = 2*(ta-tb)/(ta+tb)
            deltas.append(advantage)
            lines.append(f"| {i//2+1} | {ta:.1f} | {tb:.1f} | {advantage:+.1%} |")
        mean = sum(deltas)/len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        lines += ['', f'A{a} vs A{b}: mean advantage {mean:+.1%}, A{a} faster in {wins}/{len(deltas)} pairs.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    main()
