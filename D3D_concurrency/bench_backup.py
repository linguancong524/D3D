"""Interleaved fast/neural comparison using one model load and matched workloads."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from bench import TARGET,DRAFT,HERE,original,AutoTokenizer,Engine

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['math500'])
    p.add_argument('--concurrency',nargs='+',type=int,default=[1,8,16,32])
    p.add_argument('--max-samples',type=int,default=64)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--anchor-num',type=int,default=3)
    p.add_argument('--fast-miss-anchors',type=int,default=0)
    p.add_argument('--validate',action='store_true')
    p.add_argument('--policies',nargs='+',choices=['neural','fast','fast_legacy'],default=['neural','fast'])
    p.add_argument('--output',default=str(HERE/'results/backup_comparison/raw.json'))
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(TARGET)
    workloads={}
    for name in a.datasets:
        rows,cfg,_=original.load_rows(name,a.max_samples or None)
        workloads[name]=[tok.encode(original._apply_chat_template(tok,[{'role':'user','content':cfg['question'](r)}],False)) for r in rows]
    data={'config':vars(a),'target':TARGET,'draft':DRAFT,'hardware':[torch.cuda.get_device_name(i) for i in [0,1]],
          'metric':'actual output tokens / generation wall time including prefill; exclude load, warmup, tokenization, saving',
          'fast_semantics':'random suffix; masked anchor seed after target submission; fast uses first-accept-only miss fanout; fast_legacy retains full fanout',
          'source_sha256':{str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.rglob('*.py')},'cells':[]}
    if out.exists(): raise FileExistsError(out)
    e=Engine(TARGET,DRAFT,anchor_num=a.anchor_num,fast_miss_anchors=a.fast_miss_anchors)
    try:
        if a.validate:
            from transformers import AutoModelForCausalLM
            cases=next(iter(workloads.values()))[:4]
            e.backup='fast'
            actual=e.generate(cases,len(cases),max_new_tokens=32,ignore_eos=True)
            ref=AutoModelForCausalLM.from_pretrained(TARGET,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
            checks=[]
            for ids,result in zip(cases,actual['outputs']):
                with torch.inference_mode():
                    x=torch.tensor([ids],device='cuda:0')
                    expected=ref.generate(x,max_new_tokens=32,do_sample=False,eos_token_id=None,pad_token_id=151643)[0,len(ids):].tolist()
                checks.append({'equal':expected==result['tokens'],'expected':expected,'actual':result['tokens']})
            del ref;torch.cuda.empty_cache()
            data['validation']=checks
            data['validation_miss_requests']=actual['miss_requests']
            out.write_text(json.dumps(data,indent=2))
            if not all(c['equal'] for c in checks): raise RuntimeError('Fast greedy reference mismatch')
            print('VALIDATION PASSED',flush=True)
        for name,prompts in workloads.items():
            for c in a.concurrency:
                for backup in a.policies:
                    e.backup=backup
                    e.generate(prompts[:c],c,max_new_tokens=32)
                for repeat in range(a.repeats):
                    order=a.policies if repeat%2==0 else list(reversed(a.policies))
                    for backup in order:
                        e.backup=backup
                        print(f'RUN {name} C={c} repeat={repeat} backup={backup}',flush=True)
                        r=e.generate(prompts,c,max_new_tokens=a.max_new_tokens)
                        r.update(dataset=name,concurrency=c,repeat=repeat,requests=len(prompts),input_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest())
                        data['cells'].append(r);out.write_text(json.dumps(data,indent=2))
                        print(f'RESULT {name} C={c} {backup}: {r["tps"]:.2f} tok/s; {r["elapsed_s"]:.2f}s; misses={r["miss_requests"]}',flush=True)
    finally:e.close()

if __name__=='__main__':main()
