"""Inspect first cross-batch greedy differences without running a speed baseline.
Run separately after benchmarking so this cannot contaminate timed workloads.
"""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'D3DV2'))
import eval_sync as original

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    data=json.loads((HERE/'results/smoke_concurrency.json').read_text())
    rows,cfg,_=original.load_rows('math500',40)
    target=data['config']['target']
    tok=AutoTokenizer.from_pretrained(target)
    base=data['cells'][0]['outputs']
    cases={}
    for cell in data['cells'][1:]:
        for a,b in zip(base,cell['outputs']):
            if a['tokens']==b['tokens']:continue
            at=next(i for i,(x,y) in enumerate(zip(a['tokens'],b['tokens'])) if x!=y)
            idx=a['index'];key=(idx,at)
            cases[key]=(a['tokens'][:at],a['tokens'][at],b['tokens'][at])
    model=AutoModelForCausalLM.from_pretrained(target,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
    result=[]
    for (idx,at),(prefix,a,b) in cases.items():
        text=original._apply_chat_template(tok,[{'role':'user','content':cfg['question'](rows[idx])}],False)
        ids=tok.encode(text)+prefix
        scores=model(torch.tensor([ids],device='cuda:0'),logits_to_keep=1).logits[0,-1].float()
        values,indices=scores.topk(5)
        result.append(dict(index=idx,position=at,token_a=a,token_b=b,
                           score_a=scores[a].item(),score_b=scores[b].item(),
                           top5_tokens=indices.tolist(),top5_logits=values.tolist(),
                           text_a=tok.decode([a]),text_b=tok.decode([b])))
    (HERE/'results/batch_numerics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
