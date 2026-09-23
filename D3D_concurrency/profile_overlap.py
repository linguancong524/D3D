"""Host-wall overlap of target verification and synchronized draft prediction.

perf_counter shares a monotonic clock across local processes. Intersections
measure stage overlap, not simultaneous GPU kernel occupancy.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from bench import TARGET, DRAFT, original, AutoTokenizer, Engine


class OverlapEngine(Engine):
    def _predict(self, seqs):
        torch.cuda.synchronize(1)
        start = time.perf_counter()
        result = super()._predict(seqs)
        torch.cuda.synchronize(1)
        self.predictions.append(dict(start=start, end=time.perf_counter(),
                                     batch=len(seqs)))
        return result

    def _recover(self, seqs):
        if not seqs:
            return
        bootstrap = all(not s.accepts for s in seqs)
        torch.cuda.synchronize(1)
        start = time.perf_counter()
        result = super()._recover(seqs)
        torch.cuda.synchronize(1)
        self.recoveries.append(dict(start=start, end=time.perf_counter(),
                                    batch=len(seqs), bootstrap=bootstrap))
        return result

    def generate(self, *args, **kwargs):
        self.predictions, self.recoveries = [], []
        result = super().generate(*args, **kwargs)
        spans = self.target.request('forward_spans')
        assert len(spans) == len(self.predictions)
        rounds = []
        for prediction, (start, end) in zip(self.predictions, spans):
            overlap = max(0., min(end, prediction['end']) - max(start, prediction['start']))
            rounds.append(dict(**prediction, verify_start=start, verify_end=end,
                               draft_s=prediction['end']-prediction['start'],
                               verify_s=end-start, overlap_s=overlap,
                               draft_tail_s=max(0., prediction['end']-max(end, prediction['start']))))
        result.pop('outputs')
        result['rounds'] = rounds
        result['recoveries'] = self.recoveries
        return result


def summarize(cell, full_only=False):
    rounds = [r for r in cell['rounds'] if not full_only or r['batch'] == cell['concurrency']]
    sums = {k: sum(r[k] for r in rounds) for k in ('draft_s', 'verify_s', 'overlap_s', 'draft_tail_s')}
    return dict(rounds=len(rounds), **sums,
                draft_hidden_pct=100*sums['overlap_s']/sums['draft_s'] if sums['draft_s'] else 0,
                redraft_s=sum(r['end']-r['start'] for r in cell['recoveries'] if not r['bootstrap']),
                bootstrap_s=sum(r['end']-r['start'] for r in cell['recoveries'] if r['bootstrap']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(TARGET)
    rows, cfg, _ = original.load_rows('math500', 32)
    prompts = [tok.encode(original._apply_chat_template(tok, [{'role': 'user', 'content': cfg['question'](r)}], False)) for r in rows]
    data = dict(note='Synchronized diagnostic stage wall times, not kernel overlap or benchmark TPS. Fast prediction includes masked seed. Full-batch summary excludes drain rounds; recovery totals cover the whole run.',
                dataset='math500', requests=32, max_new_tokens=256, cells=[])
    engine = OverlapEngine(TARGET, DRAFT)
    try:
        engine.target.request('profile_forward', True)
        for concurrency in (16, 32):
            for anchor, policy, miss in ((3, 'neural', 0), (3, 'fast', 3),
                                         (1, 'neural', 0), (1, 'fast', 1), (1, 'fast', 16)):
                engine.A, engine.backup, engine.fast_miss_anchors = anchor, policy, miss
                engine.generate(prompts, concurrency, max_new_tokens=32)
                cell = engine.generate(prompts, concurrency, max_new_tokens=256)
                cell.update(concurrency=concurrency, anchor_num=anchor, fast_miss_anchors=miss)
                cell['summary'] = summarize(cell)
                cell['full_batch_summary'] = summarize(cell, True)
                data['cells'].append(cell)
                args.output.write_text(json.dumps(data, indent=2))
                print(concurrency, anchor, policy, miss, cell['summary'], 'full', cell['full_batch_summary'], flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
