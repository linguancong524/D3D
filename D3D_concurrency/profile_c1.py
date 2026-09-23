"""C1 fixed-workload latency/attention ablations, separate from throughput runs."""
import os
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import sys,json,time,statistics
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.profiler import profile,ProfilerActivity,record_function
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'D3DV2'))
from bench import TARGET,DRAFT
from thread_engine_reference import Engine
from backend import Sequence,SamplingParams
from transformers import AutoTokenizer
from flash_attn import flash_attn_with_kvcache
from torch.nn.functional import scaled_dot_product_attention as sdpa
import eval_sync as original

OUT=HERE/'results/aime_diagnosis/profile.json'
OLD=HERE.parent/'D3DV2/results_pruning/qwen3_p1.7b_to_8b_d24_d28/raw/d28/2026_09_14_08_23_47/aime25/aime25_d28_with_Qwen3-8B_anchor3.json'

def measure(fn,devices=(0,1),n=20,warmup=3):
    for _ in range(warmup):fn()
    vals=[]
    for _ in range(n):
        for d in devices:torch.cuda.synchronize(d)
        t=time.perf_counter();fn()
        for d in devices:torch.cuda.synchronize(d)
        vals.append((time.perf_counter()-t)*1000)
    return dict(median_ms=statistics.median(vals),mean_ms=statistics.mean(vals),min_ms=min(vals),repeats=n)

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    engine=Engine(TARGET,DRAFT,pages=256)
    tokenizer=AutoTokenizer.from_pretrained(TARGET)
    old=json.loads(OLD.read_text())
    row=next(p for p in old['prompts'] if p['output_tokens']==2048)
    ids=tokenizer.encode(original._apply_chat_template(tokenizer,[{'role':'user','content':row['prompt']}],False))
    continuation=[t for step in row['steps'] for t in step['emitted_tokens']]
    full=ids+continuation
    results={'prefixes':{},'attention':{}}
    try:
        for length in [256,1024,2048]:
            s=Sequence(full[:length],SamplingParams(temperature=0));s.prefix=length;s.recoveries=0
            s.anchor=engine.target.prefill([s])[0];engine.draft.prefill([s]);engine._recover([s])
            block=[s.current.cpu().tolist()]
            def target():return engine._verify([s],block)
            def draft():return engine._predict([s])
            def overlap():
                f=engine.executor.submit(target);draft();return f.result()
            r={}
            r['target_alone']=measure(target)
            r['packed_draft_alone']=measure(draft)
            r['threaded_overlap']=measure(overlap)
            saved={}
            original_forward=engine.draft.forward
            def capture(*args,**kwargs):
                saved['args']=args;saved['kwargs']=kwargs
                return original_forward(*args,**kwargs)
            engine.draft.forward=capture
            draft()
            engine.draft.forward=original_forward
            r['packed_forward_only']=measure(lambda:original_forward(*saved['args'],**saved['kwargs']))
            r['recovery_alone']=measure(lambda:engine._recover([s]))
            # Profile fixed packed work; profiler data are NOT throughput results.
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    with record_function('packed_draft'):draft()
                torch.cuda.synchronize(1)
            r['draft_cuda_ops']=prof.key_averages().table(sort_by='self_device_time_total',row_limit=15)
            r['draft_cpu_ops']=prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=15)
            # F.linear: exact same head and hidden workload, only split size changes.
            hidden=torch.randn(736,engine.draft.config.hidden_size,device='cuda:1',dtype=torch.bfloat16)
            weight=engine.draft.model.lm_head.weight
            def head(chunk):
                for h in hidden.split(chunk):
                    logits=torch.nn.functional.linear(h,weight);logits.argmax(-1);logits.topk(4,dim=-1)
            r['head_chunk256']=measure(lambda:head(256),devices=(1,),n=30)
            r['head_chunk736']=measure(lambda:head(736),devices=(1,),n=30)
            results['prefixes'][str(length)]=r
            print('PREFIX',length,{k:v for k,v in r.items() if not k.endswith('_ops')},flush=True)
            engine.target.release(s);engine.draft.release(s)
            OUT.write_text(json.dumps(results,indent=2))
        # Isolated target attention: no model/scheduler changes, B=1,Q=16,H=32,KVH=8,D=128.
        for length in [256,1024,2048]:
            torch.cuda.set_device(0);dev='cuda:0';S=length+16
            pages=(S+255)//256
            kc=torch.randn(pages,256,8,128,device=dev,dtype=torch.bfloat16)
            vc=torch.randn_like(kc);q=torch.randn(1,16,32,128,device=dev,dtype=torch.bfloat16)
            bt=torch.arange(pages,device=dev,dtype=torch.int32)[None]
            lens=torch.tensor([S],device=dev,dtype=torch.int32)
            k=kc.view(-1,8,128)[:S].transpose(0,1)[None].repeat_interleave(4,dim=1)
            v=vc.view(-1,8,128)[:S].transpose(0,1)[None].repeat_interleave(4,dim=1)
            mask=torch.arange(S,device=dev)[None,:] <= (length+torch.arange(16,device=dev))[:,None]
            def fa():return flash_attn_with_kvcache(q,kc,vc,cache_seqlens=lens,block_table=bt,causal=True)
            def sa():return sdpa(q.transpose(1,2),k,v,attn_mask=mask)
            a=fa().transpose(1,2);b=sa()
            results['attention'][str(length)]={'paged_flash':measure(fa,devices=(0,),n=100),
                    'dense_sdpa_kernel_only':measure(sa,devices=(0,),n=100),
                    'max_abs_difference':(a-b).abs().max().item()}
        OUT.write_text(json.dumps(results,indent=2))
    finally:engine.close()

if __name__=='__main__':main()
