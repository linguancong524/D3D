"""Audit and summarize a completed paired backup experiment."""
import argparse,itertools,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');a=p.parse_args()
path=Path(a.input);data=json.loads(path.read_text());cfg=data['config']
expected=set(itertools.product(cfg['datasets'],cfg['concurrency'],range(cfg['repeats']),['neural','fast']))
seen=set();hashes={}
for c in data['cells']:
    key=(c['dataset'],c['concurrency'],c['repeat'],c['backup'])
    assert key not in seen;seen.add(key)
    assert len(c['outputs'])==c['requests']
    assert [o['index'] for o in c['outputs']]==list(range(c['requests']))
    assert c['target_active_sequences']==0 and c['target_free_pages']==512
    assert sum(len(o['tokens']) for o in c['outputs'])==c['output_tokens']
    assert abs(c['output_tokens']/c['elapsed_s']-c['tps'])<1e-9
    assert c['max_active']==max(c['decode_batch_sizes'])<=c['concurrency']
    assert hashes.setdefault(c['dataset'],c['input_sha256'])==c['input_sha256']
    for o in c['outputs']:
        assert 0<len(o['tokens'])<=cfg['max_new_tokens']
        assert not any(t in (151643,151645) for t in o['tokens'][:-1])
    if c['backup']=='fast':
        assert sum(o['fast_backups'] for o in c['outputs'])==c['miss_requests']
        assert sum(o['fast_seeds'] for o in c['outputs'])==c['miss_requests']
        assert all(o['recoveries']<=1 for o in c['outputs'])
    else:assert not any(o['fast_backups'] for o in c['outputs'])
assert seen==expected,(expected-seen)
rows=[]
for name in cfg['datasets']:
    for con in cfg['concurrency']:
        row={'dataset':name,'concurrency':con}
        for mode in ['neural','fast']:
            runs=[c for c in data['cells'] if c['dataset']==name and c['concurrency']==con and c['backup']==mode]
            toks=sum(c['output_tokens'] for c in runs);secs=sum(c['elapsed_s'] for c in runs)
            accepts=[v for c in runs for o in c['outputs'] for v in o['accepts']]
            rounds=sum(len(c['decode_batch_sizes']) for c in runs)
            row[mode]={'tps':toks/secs,'run_tps':[c['tps'] for c in runs],
                'output_tokens':toks,'elapsed_s':secs,'request_steps':len(accepts),
                'mean_accepted':sum(accepts)/len(accepts),'effective_output_per_step':toks/len(accepts),
                'miss_requests':sum(c['miss_requests'] for c in runs),
                'miss_round_fraction':sum(c['miss_rounds'] for c in runs)/rounds,
                'scheduler_rounds':rounds}
        row['fast_over_neural']=row['fast']['tps']/row['neural']['tps']
        agreements=[]
        for repeat in range(cfg['repeats']):
            n=next(c for c in data['cells'] if (c['dataset'],c['concurrency'],c['repeat'],c['backup'])==(name,con,repeat,'neural'))
            f=next(c for c in data['cells'] if (c['dataset'],c['concurrency'],c['repeat'],c['backup'])==(name,con,repeat,'fast'))
            agreements.append(sum(x['tokens']==y['tokens'] for x,y in zip(n['outputs'],f['outputs'])))
        row['identical_output_requests_per_repeat']=agreements
        rows.append(row)
path.with_name('summary.json').write_text(json.dumps(rows,indent=2))
lines=['# Fast vs Neural Backup (SDAR adaptation)','',
       f"Datasets: {cfg['datasets']}; first {cfg['max_samples']} prompts; max output {cfg['max_new_tokens']}; {cfg['repeats']} repeats. Natural EOS; greedy; 2 H800; SDAR-1.7B → Qwen3-8B.",
       f"One model load; alternating policy order. Same prompts and caches; anchor={cfg.get('anchor_num',3)}, fast miss anchors={cfg.get('fast_miss_anchors',0)} (0=same as anchor); draft microbatch 4. Timing includes prefill and all recovery/seed work.",
       '', '| Dataset | C | Neural TPS | Fast TPS | Fast/Neural | Neural tokens/step | Fast tokens/step |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['dataset']} | {r['concurrency']} | {r['neural']['tps']:.1f} | {r['fast']['tps']:.1f} | {r['fast_over_neural']:.3f}× | {r['neural']['effective_output_per_step']:.2f} | {r['fast']['effective_output_per_step']:.2f} |")
lines+=['','TPS pooled over repetitions. Ratios compare backup policies, not an AR baseline.',
        'Fast is a greedy SDAR adaptation: random in-vocabulary suffixes are returned on misses; next-outcome masked anchor seeding runs after target submission. Initial admission uses the same neural bootstrap in both policies.',
        'SSD reference uses AR glue-decode to build next-outcome predictions and random-logit argmax for miss tokens; this code uses SDAR masked seeding and uniform integer random tokens. It does not reproduce SSD AR kernels or its full scheduler. Measured fast semantics: '+data.get('fast_semantics','See source hashes.'),
        'Exact output agreement and acceptance/miss statistics are in summary.json. BF16 is not promised bitwise identical across different speculative trajectories. These are capped subset performance measurements, not a full accuracy evaluation.','']
path.with_name('table.md').write_text('\n'.join(lines));print('\n'.join(lines))
