#!/usr/bin/env python3
"""CPU-only integrity checks for measured throughput artifacts."""
import argparse
import itertools
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('input');a=p.parse_args()
data=json.loads(Path(a.input).read_text());cfg=data['config']
expected=set(itertools.product(cfg['datasets'],cfg['concurrency'],range(cfg['repeats'])))
seen=set();hashes={}
for cell in data['cells']:
    key=(cell['dataset'],cell['concurrency'],cell['repeat'])
    assert key not in seen,f'duplicate {key}'
    seen.add(key)
    assert cell['elapsed_s']>0
    assert len(cell['outputs'])==cell['requests']
    assert sorted(o['index'] for o in cell['outputs'])==list(range(cell['requests']))
    total=sum(len(o['tokens']) for o in cell['outputs'])
    assert total==cell['output_tokens']
    assert abs(total/cell['elapsed_s']-cell['tps'])<1e-9
    assert 1<=cell['max_active']<=cell['concurrency']
    assert all(1<=b<=cell['concurrency'] for b in cell['decode_batch_sizes'])
    assert cell['max_active']==max(cell['decode_batch_sizes'])
    for o in cell['outputs']:
        assert 1<=len(o['tokens'])<=cfg['max_new_tokens']
        if not cfg['ignore_eos']:
            assert not any(t in (151643,151645) for t in o['tokens'][:-1])
        assert o['latency_s']>0
    old=hashes.setdefault(cell['dataset'],cell['input_sha256'])
    assert old==cell['input_sha256'],'different workloads across concurrency levels'
assert seen==expected,f'missing {expected-seen}; unexpected {seen-expected}'
print(f'PASS: {len(seen)} measured cells, token counts, wall-time TPS, admission caps, request identities, EOS and workload hashes.')
