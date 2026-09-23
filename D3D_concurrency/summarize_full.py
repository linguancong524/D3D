#!/usr/bin/env python3
"""Write the results_full CSV summaries.

Per cell the reported result is the HIGH value: the attempt with the largest
TPS. The other columns of that row come from the same attempt, so the row stays
internally consistent. `tps_attempts` keeps every attempt and `tps_pooled`
keeps total tokens over total wall time, so the spread and the pooling view are
never hidden.

When a dataset has more than one run (the user-requested math500 re-measurement
superseding the first pass), the newest run that has the cell complete wins;
the `run_id` and `origin` columns say which one the row came from.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results_full'
FIELDS = ['pair', 'dataset', 'concurrency', 'anchor', 'microbatch', 'attempts', 'tps_best',
          'tps_attempts', 'tps_pooled', 'decode_tps_best', 'hit_rate', 'mean_accept',
          'output_tokens_best', 'wall_s_best', 'prefill_s_best', 'peak_target_gb', 'run_id', 'origin']


def load_runs(pair, dataset):
    """All complete-enough runs for one dataset, oldest first."""
    runs = []
    base = ROOT / pair / dataset
    if not base.exists():
        return runs
    for directory in sorted(path for path in base.iterdir() if path.is_dir()):
        benchmark = directory / 'benchmark.json'
        if benchmark.exists():
            runs.append((directory.name, json.loads(benchmark.read_text())))
    return runs


def cell_rows(pair, dataset):
    runs = load_runs(pair, dataset)
    if not runs:
        return []
    first_run = runs[0][0]
    concurrencies = sorted({c['concurrency'] for _, data in runs for c in data['cells']})
    rows = []
    for concurrency in concurrencies:
        cells, run_id = [], None
        for candidate_id, data in reversed(runs):  # newest first
            attempts = [c for c in data['cells'] if c['concurrency'] == concurrency]
            if len(attempts) >= min(2, data['config']['repeats']) or (attempts and data.get('status') != 'running'):
                cells, run_id = attempts, candidate_id
                break
        if not cells:
            continue
        best = max(cells, key=lambda c: c['tps'])
        tokens = sum(c['output_tokens'] for c in cells)
        wall = sum(c['elapsed_s'] for c in cells)
        rows.append({
            'pair': pair, 'dataset': dataset, 'concurrency': concurrency,
            'anchor': best['anchor_num'], 'microbatch': best['draft_microbatch'],
            'attempts': len(cells),
            'tps_best': round(best['tps'], 2),
            'tps_attempts': ', '.join(f"{c['tps']:.1f}" for c in cells),
            'tps_pooled': round(tokens/wall, 2),
            'decode_tps_best': round(best['decode_tps'], 2) if best.get('decode_tps') else '',
            'hit_rate': f"{best['hit_rate']:.4f}",
            'mean_accept': round(best['mean_accept'], 3),
            'output_tokens_best': best['output_tokens'],
            'wall_s_best': round(best['elapsed_s'], 1),
            'prefill_s_best': round(best['prefill_s'], 1) if best.get('prefill_s') else '',
            'peak_target_gb': round(best['peak_allocated_gb'][0], 2),
            'run_id': run_id, 'origin': 'first pass' if run_id == first_run else 're-measurement',
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    pairs = sorted(path.name for path in ROOT.iterdir() if path.is_dir() and path.name.startswith('1.7b_'))
    everything = []
    for pair in pairs:
        datasets = sorted(path.name for path in (ROOT / pair).iterdir() if path.is_dir())
        pair_rows = []
        for dataset in datasets:
            rows = cell_rows(pair, dataset)
            if not rows:
                continue
            write_csv(ROOT / pair / dataset / 'summary.csv', rows)
            pair_rows.extend(rows)
        if pair_rows:
            write_csv(ROOT / pair / 'summary.csv', pair_rows)
            everything.extend(pair_rows)
        print(f'{pair}: {len(pair_rows)} cells -> {ROOT/pair/"summary.csv"}')
    if everything:
        write_csv(ROOT / 'summary.csv', everything)
        (ROOT / 'table.txt').write_text(render_table(everything))
        (ROOT / 'table.md').write_text(render_table(everything, markdown=True))
        print(f'combined: {len(everything)} cells -> {ROOT/"summary.csv"}')
        print(render_table(everything))


# Curated baseline tables, not a directory scan (results_baseline also holds
# smoke_* runs and user-cancelled reruns). The user designated the 32B
# summary_n32_t512.csv as the comparison baseline; its tps_source column says
# which rows are measured and which are substituted.
BASELINE_CSV = {'8B': 'summary_n32_t512.csv', '32B': 'summary_n32_t512.csv'}


def baseline_tps(pair, dataset):
    """(pooled AR baseline TPS per concurrency, tps_source note) for a dataset."""
    size = '32B' if '32b' in pair else '8B'
    path = HERE / 'results_baseline' / size / BASELINE_CSV[size]
    if not path.exists():
        return {}, ''
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['benchmark'] == dataset:
                return ({int(key.rsplit('_', 2)[1]): float(value)
                         for key, value in row.items() if key.startswith('concurrency_')},
                        row.get('tps_source', '') or '')
    return {}, ''


def render_table(rows, markdown=False):
    """One line per dataset, four concurrencies, TPS with in-parenthesis speedup."""
    titles = {'math500': 'Math500', 'humaneval': 'HumanEval', 'mt-bench': 'MT-Bench'}
    pairs = sorted({row['pair'] for row in rows}, key=lambda p: (1 if '32b' in p else 0, p))
    concurrencies = sorted({row['concurrency'] for row in rows})
    labels = {1: '1.7B -> 8B (Qwen3-8B)', 2: '1.7B -> 32B (Qwen3-32B)'}
    lines = [f"Concurrency\t" + '\t'.join(str(c) for c in concurrencies)]
    missing_baseline = []
    mixed_origin = []
    substituted = []
    for pair in pairs:
        lines += ['', labels.get(1 if pair.endswith('8b') else 2, pair)]
        for dataset in sorted({row['dataset'] for row in rows if row['pair'] == pair}):
            cells = {row['concurrency']: row for row in rows
                     if row['pair'] == pair and row['dataset'] == dataset}
            base, source_note = baseline_tps(pair, dataset)
            if not base:
                missing_baseline.append(f'{pair} {dataset}')
            if 'substituted' in source_note:
                substituted.append(f'{pair} {dataset}')
            if any(cell['origin'] == 'first pass' for cell in cells.values()):
                mixed_origin.append(f'{pair} {dataset}')
            lines.append(titles.get(dataset, dataset))
            fields = []
            for concurrency in concurrencies:
                cell = cells.get(concurrency)
                if cell is None:
                    fields.append('')
                    continue
                speedup = base.get(concurrency)
                fields.append(f"{cell['tps_best']:.0f} ({cell['tps_best']/speedup:.1f}x)"
                              if speedup else f"{cell['tps_best']:.0f} (n/a)")
            lines.append('\t'.join(fields))
    lines += ['', 'Cells are the high value (fastest attempt) of the cell; the value in parentheses is that',
              'TPS divided by the AR baseline TPS of the same verifier at the same concurrency (8B: pooled',
              'measurement; 32B: quiet-node estimate, `results_baseline/32B/summary_n32_t512.csv`).']
    if missing_baseline:
        lines.append('No AR baseline yet (speedup shown as n/a): ' + '; '.join(missing_baseline) + '.')
    if mixed_origin:
        lines.append('Still first-pass numbers, measured while the host was contended: '
                     + '; '.join(mixed_origin) + '.')
    if substituted:
        lines.append('Baseline rows carrying tps_source=estimated/substituted (not measured for that dataset): '
                     + '; '.join(substituted) + '.')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
