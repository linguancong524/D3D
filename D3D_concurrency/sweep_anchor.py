"""Small Neural anchor sweep with full-concurrency draft microbatches."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from bench import TARGET, DRAFT, original, AutoTokenizer, Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(TARGET)
    rows, cfg, source = original.load_rows('aime25', 16)
    prompts = [tok.encode(original._apply_chat_template(tok, [{'role': 'user', 'content': cfg['question'](r)}], False)) for r in rows]
    data = dict(dataset='aime25', requests=len(prompts), max_new_tokens=512,
                target=TARGET, draft=DRAFT, backup='neural',
                metric='output tokens / wall seconds including prefill and bootstrap; excludes warmup and model loading',
                input_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                hardware=[torch.cuda.get_device_name(i) for i in (0, 1)], cells=[])
    engine = Engine(TARGET, DRAFT)
    try:
        for concurrency in (16, 8, 4, 1):
            for anchor in (3, 2, 1):
                engine.A, engine.micro = anchor, concurrency
                print(f'WARMUP C={concurrency} A={anchor} micro={engine.micro}', flush=True)
                engine.generate(prompts[:concurrency], concurrency, max_new_tokens=32)
                print(f'RUN C={concurrency} A={anchor} micro={engine.micro}', flush=True)
                result = engine.generate(prompts, concurrency, max_new_tokens=512)
                hits = sum(o['hits'] for o in result['outputs'])
                result.update(concurrency=concurrency, anchor_num=anchor,
                              draft_microbatch=engine.micro,
                              hit_rate=hits/(hits+result['miss_requests']) if hits+result['miss_requests'] else 0)
                data['cells'].append(result)
                args.output.write_text(json.dumps(data, indent=2))
                lines = ['| Concurrency | Anchor | Microbatch | TPS (incl. prefill) | Hit rate | Draft peak GB |',
                         '|---:|---:|---:|---:|---:|---:|']
                for c in data['cells']:
                    lines.append(f"| {c['concurrency']} | {c['anchor_num']} | {c['draft_microbatch']} | {c['tps']:.2f} | {c['hit_rate']:.1%} | {c['peak_allocated_gb'][1]:.2f} |")
                args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
                print(f"RESULT C={concurrency} A={anchor} TPS={result['tps']:.2f} hit={result['hit_rate']:.3f} peak={result['peak_allocated_gb']}", flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
