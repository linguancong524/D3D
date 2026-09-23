#!/usr/bin/env python3
"""Render only measured cells, never infer missing throughput numbers."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output',default=None)
a=p.parse_args();source=Path(a.input);data=json.loads(source.read_text())
out=Path(a.output) if a.output else source.with_suffix('.md')
names=data['config']['datasets'];concurrencies=data['config']['concurrency']
labels={'math500':'Math500','humaneval':'HumanEval','mt-bench':'MT-Bench'}
lines=['# D3DV2 concurrency throughput: SDAR-1.7B → Qwen3-8B','',
       'Each cell reports aggregate output throughput in tok/s. No AR baseline or speedup comparison.','',
       '| Concurrency | '+' | '.join(labels.get(n,n) for n in names)+' |',
       '|---:|'+'---:|'*len(names)]
rows=[]
for c in concurrencies:
    row={'Concurrency':c};cells=[]
    for n in names:
        runs=[r for r in data['cells'] if r['concurrency']==c and r['dataset']==n]
        value=sum(r['output_tokens'] for r in runs)/sum(r['elapsed_s'] for r in runs) if runs else None
        cells.append(f'{value:.1f}' if value is not None else '—')
        row[labels.get(n,n)]=value
    rows.append(row);lines.append('| '+str(c)+' | '+' | '.join(cells)+' |')
cfg=data['config']
lines+=['',f"Initial measurement: first {cfg['max_samples']} prompts per dataset." if cfg['max_samples'] else 'All available prompts per dataset.',f"Hardware: {', '.join(data['hardware'])}; target on GPU0, draft on GPU1; BF16.",
        f"Greedy; thinking disabled; block size 16; anchor count {cfg.get('anchor_num',3)}; draft microbatch {cfg['draft_microbatch']}; backup {cfg.get('backup','neural')}; fast miss anchors {cfg.get('fast_miss_anchors',0)} (0=same as anchor count).",
        f"Maximum output length: {cfg['max_new_tokens']} tokens; ignore EOS: {cfg['ignore_eos']}; repeats per cell: {cfg['repeats']}.",
        'MT-Bench uses the first turn, matching the existing D3DV2 loader.',
        'TPS = actual emitted output tokens / full generation wall time, including prefill and final drain; loading, warmup, tokenization and file output excluded.',
        'Repeated measurements are pooled: sum(output tokens) / sum(elapsed seconds).',
        'Concurrency is an admission cap; at most C requests are active. Completed requests are replaced at the next scheduler iteration.',
        'This is a custom SSD-derived engine with SDAR packed attention, not vLLM.',
        '', 'Measured workload:', '', '| Dataset | Requests | Output tokens (C=1) |','|---|---:|---:|']
for n in names:
    runs=[r for r in data['cells'] if r['dataset']==n and r['concurrency']==1]
    if runs:lines.append(f"| {labels.get(n,n)} | {runs[0]['requests']} | {runs[0]['output_tokens']} |")
lines+=['',f'Raw measurements: `{source.name}`.','']
out.write_text('\n'.join(lines))
with out.with_suffix('.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print('\n'.join(lines))

tex=[r'\begin{table}[t]',r'\centering',
     r'\caption{D3DV2 output throughput (tok/s), SDAR-1.7B $\rightarrow$ Qwen3-8B, two H800 GPUs. '+
     f"{cfg['max_samples'] or 'All available'} prompts per dataset, at most {cfg['max_new_tokens']} output tokens, "+
     f"{cfg['repeats']} measurement(s) per cell. MT-Bench uses first turns. "+
     r'Timing includes prefill and decode.}',
     r'\label{tab:d3dv2-concurrency}',r'\begin{tabular}{r'+('r'*len(names))+'}',r'\hline',
     'Concurrency & '+' & '.join(labels.get(n,n) for n in names)+r' \\',r'\hline']
for row in rows:
    vals=[row[labels.get(n,n)] for n in names]
    tex.append(str(row['Concurrency'])+' & '+' & '.join(f'{v:.1f}' if v is not None else '--' for v in vals)+r' \\')
tex += [r'\hline',r'\end{tabular}',r'\end{table}','']
out.with_suffix('.tex').write_text('\n'.join(tex))
