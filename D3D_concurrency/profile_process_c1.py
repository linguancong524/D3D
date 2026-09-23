"""Fixed-state process overlap control; does not modify production engine."""
import os
os.environ.setdefault('TORCH_COMPILE_DISABLE','1')
import sys,json,multiprocessing as mp
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'D3DV2'))
from profile_c1 import measure,OLD
from bench import TARGET,DRAFT
from backend import Runner,Sequence,SamplingParams
from engine import Engine
from transformers import AutoTokenizer
import eval_sync as original

def worker(pipe):
    torch.set_num_threads(4)
    try:
        with torch.inference_mode():
            runner=Runner(TARGET,'cuda:0',pages=256)
            s=None
            pipe.send(('ready',None))
            while True:
                command,payload=pipe.recv()
                if command=='stop':break
                if command=='prefill':
                    if s is not None:runner.release(s)
                    s=Sequence(payload,SamplingParams(temperature=0));s.prefix=len(payload)
                    pipe.send(('ok',runner.prefill([s])[0]))
                elif command=='verify':
                    block=payload
                    post=runner.forward([s],[block]).cpu().tolist()[0]
                    n=1
                    while n<len(block) and block[n]==post[n-1]:n+=1
                    pipe.send(('ok',(n,post[n-1])))
    except BaseException:
        import traceback
        pipe.send(('error',traceback.format_exc()))
        raise

def receive(pipe):
    kind,value=pipe.recv()
    if kind=='error':raise RuntimeError(value)
    return value

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    ctx=mp.get_context('spawn');p1,p2=ctx.Pipe()
    process=ctx.Process(target=worker,args=(p2,));process.start();p2.close()
    try:
        engine=Engine.__new__(Engine)
        engine.K=16;engine.A=3;engine.micro=4
        engine.backup='neural';engine.fast_miss_anchors=0
        engine.draft=Runner(DRAFT,'cuda:1',draft=True,pages=256)
        receive(p1)
        tokenizer=AutoTokenizer.from_pretrained(TARGET)
        row=next(p for p in json.loads(OLD.read_text())['prompts'] if p['output_tokens']==2048)
        ids=tokenizer.encode(original._apply_chat_template(tokenizer,[{'role':'user','content':row['prompt']}],False))
        full=ids+[t for step in row['steps'] for t in step['emitted_tokens']]
        results={}
        for length in [256,1024,2048]:
            s=Sequence(full[:length],SamplingParams(temperature=0));s.prefix=length;s.recoveries=0
            p1.send(('prefill',s.token_ids));s.anchor=receive(p1)
            engine.draft.prefill([s]);engine._recover([s]);block=s.current.cpu().tolist()
            def target():p1.send(('verify',block));return receive(p1)
            def draft():return engine._predict([s])
            def overlap():p1.send(('verify',block));draft();return receive(p1)
            results[str(length)]={
                'target_with_ipc':measure(target,devices=(1,)),
                'draft_alone':measure(draft,devices=(1,)),
                'process_overlap':measure(overlap,devices=(1,))}
            engine.draft.release(s)
            print(length,results[str(length)],flush=True)
            (HERE/'results/aime_diagnosis/process_profile.json').write_text(json.dumps(results,indent=2))
    finally:
        if process.is_alive():p1.send(('stop',None));process.join(timeout=30)
        if process.is_alive():process.terminate();process.join()
        p1.close()
    if process.exitcode:raise RuntimeError(f'worker exit {process.exitcode}')

if __name__=='__main__':main()
