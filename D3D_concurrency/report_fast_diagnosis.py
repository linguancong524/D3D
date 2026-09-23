"""Audit paired backup runs and report accepted work and wall cost separately."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('inputs',nargs='+',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
lines=['# Fast Backup diagnosis','',
       'Pooled actual output tokens / wall time including prefill, natural EOS. '
       'Ratios compare backup policies, not AR speedup. SDAR-1.7B -> Qwen3-8B.', '',
       '| Dataset | Cap | Samples | Anchor | C | Policy | TPS | vs Neural | Tokens/request-step | ms/request-step | Misses |',
       '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
sources=[]
for path in a.inputs:
    data=json.loads(path.read_text()); cfg=data['config']
    policies=cfg.get('policies',['neural','fast'])
    seen=set()
    hashes={}
    for c in data['cells']:
        key=(c['dataset'],c['concurrency'],c['repeat'],c['backup'])
        assert key not in seen; seen.add(key)
        assert len(c['outputs'])==c['requests']
        assert c['target_active_sequences']==0 and c['target_free_pages']==512
        assert sum(len(o['tokens']) for o in c['outputs'])==c['output_tokens']
        assert c['max_active']<=c['concurrency']
        assert hashes.setdefault(c['dataset'],c['input_sha256'])==c['input_sha256']
        assert all(0<len(o['tokens'])<=cfg['max_new_tokens'] for o in c['outputs'])
    assert len(seen)==len(cfg['datasets'])*len(cfg['concurrency'])*cfg['repeats']*len(policies)
    for ds in cfg['datasets']:
        for con in cfg['concurrency']:
            grouped={policy:[c for c in data['cells'] if c['dataset']==ds and c['concurrency']==con and c['backup']==policy]
                     for policy in policies}
            baseline=grouped['neural']
            base=sum(c['output_tokens'] for c in baseline)/sum(c['elapsed_s'] for c in baseline)
            for policy,cells in grouped.items():
                label=policy
                if policy=='fast': label+=f'(miss={cfg.get("fast_miss_anchors",0) or cfg.get("anchor_num",3)})'
                tokens=sum(c['output_tokens'] for c in cells)
                secs=sum(c['elapsed_s'] for c in cells)
                steps=sum(len(o['accepts']) for c in cells for o in c['outputs'])
                misses=sum(c['miss_requests'] for c in cells)
                lines.append(f'| {ds} | {cfg["max_new_tokens"]} | {cfg["max_samples"]} | {cfg.get("anchor_num",3)} | {con} | {label} | {tokens/secs:.1f} | {tokens/secs/base:.3f} | {tokens/steps:.2f} | {1000*secs/steps:.2f} | {misses} |')
    sources.append(f'Source: `{path}`; repeats={cfg["repeats"]}; fast miss anchors={cfg.get("fast_miss_anchors",0)} (0 means same as anchor).')
lines.extend(['']+sources)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
