#!/usr/bin/env python3
import argparse
import faulthandler
faulthandler.enable()

import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'D3DV2'))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import AutoTokenizer
from engine import Engine
import eval_sync as original

TARGET=str(ROOT.parent/'models/Qwen3-8B')
DRAFT='/inspire/ssd/project/sais-bio/public/meiruiqiao/qwen-bard-opd/sdar_output/sdar_1_7/ar1_b16_a128_from355227_ce_2epoch_2gpu/step-1420908'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',default=TARGET);p.add_argument('--draft',default=DRAFT)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,8,16,32])
    p.add_argument('--datasets',nargs='+',default=['math500','humaneval','mt-bench'])
    p.add_argument('--max-samples',type=int,default=0)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--draft-microbatch',type=int,default=4)
    p.add_argument('--backup',choices=['neural','fast','fast_legacy'],default='neural')
    p.add_argument('--anchor-num',type=int,default=3)
    p.add_argument('--fast-miss-anchors',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pages',type=int,default=512)
    p.add_argument('--ignore-eos',action='store_true')
    p.add_argument('--output',default=str(HERE/'results/benchmark.json'))
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(a.target)
    datasets={}
    for name in a.datasets:
        rows,cfg,source=original.load_rows(name,a.max_samples or None)
        texts=[original._apply_chat_template(tok,[{'role':'user','content':cfg['question'](r)}],False) for r in rows]
        datasets[name]=[tok.encode(t) for t in texts]
    engine=Engine(a.target,a.draft,draft_microbatch=a.draft_microbatch,pages=a.pages,backup=a.backup,seed=a.seed,anchor_num=a.anchor_num,fast_miss_anchors=a.fast_miss_anchors)
    payload={'config':vars(a),'hardware':[torch.cuda.get_device_name(i) for i in [0,1]],
             'torch':torch.__version__,'ssd_commit':'d7eb8fa0edb77a6d0876af1903367b9bb82f54e7',
             'metric':'actual output tokens / wall time including prefill, excluding model load/warmup/tokenization',
             'enable_thinking':False,'temperature':0,'block_size':16,'anchor_num':a.anchor_num,
             'torch_compile':False,'target_execution':'persistent_process','source_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.rglob('*.py')},'cells':[]}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    try:
        first=next(iter(datasets.values()))
        print('WARMUP',flush=True)
        engine.generate(first[:min(2,len(first))],2,max_new_tokens=32)
        if a.validate:
            # Check batched target verification against independent greedy target decoding.
            from transformers import AutoModelForCausalLM
            ref=AutoModelForCausalLM.from_pretrained(a.target,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
            validation=[]
            cases=first[:min(4,len(first))]
            actual=engine.generate(cases,len(cases),max_new_tokens=32,ignore_eos=True)
            for ids,res in zip(cases,actual['outputs']):
                x=torch.tensor([ids],device='cuda:0')
                with torch.inference_mode():
                    y=ref.generate(x,max_new_tokens=32,do_sample=False,eos_token_id=None,pad_token_id=151643)
                expected=y[0,len(ids):].tolist()
                same=expected==res['tokens']
                validation.append({'index':res['index'],'equal':same,'expected':expected,'actual':res['tokens']})
            del ref;torch.cuda.empty_cache()
            payload['validation']=validation
            out.write_text(json.dumps(payload,indent=2))
            if not all(v['equal'] for v in validation): raise RuntimeError('Greedy reference mismatch; inspect validation JSON')
            from validation import validate_draft
            payload['draft_validation']=validate_draft(engine,a.draft,first[0])
            out.write_text(json.dumps(payload,indent=2))
            print('VALIDATION PASSED',flush=True)
        for name,prompts in datasets.items():
            for c in a.concurrency:
                print(f'WARMUP dataset={name} C={c}',flush=True)
                engine.generate(prompts[:c],c,max_new_tokens=32,ignore_eos=a.ignore_eos)
                for repeat in range(a.repeats):
                    print(f'RUN dataset={name} C={c} repeat={repeat}',flush=True)
                    result=engine.generate(prompts,c,max_new_tokens=a.max_new_tokens,ignore_eos=a.ignore_eos)
                    result.update(dataset=name,concurrency=c,repeat=repeat,requests=len(prompts),
                                  input_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest())
                    payload['cells'].append(result)
                    out.write_text(json.dumps(payload,indent=2))
                    print(f'RESULT {name} C={c}: {result["tps"]:.2f} tok/s, {result["elapsed_s"]:.2f}s, max_active={result["max_active"]}',flush=True)
        print('DONE',out,flush=True)
    finally: engine.close()

if __name__=='__main__':main()
