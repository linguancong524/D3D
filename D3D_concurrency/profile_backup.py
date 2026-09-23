"""Synchronized stage accounting; diagnostic numbers, not throughput claims."""
import argparse
import json
import time
from pathlib import Path
import torch
from bench import TARGET,DRAFT,original,AutoTokenizer,Engine


class ProfileEngine(Engine):
    def _predict(self,seqs):
        torch.cuda.synchronize(1)
        start=time.perf_counter()
        result=super()._predict(seqs)
        torch.cuda.synchronize(1)
        self.times['prediction_s']+=time.perf_counter()-start
        return result

    def _recover(self,seqs):
        torch.cuda.synchronize(1)
        start=time.perf_counter()
        result=super()._recover(seqs)
        torch.cuda.synchronize(1)
        self.times['recovery_s']+=time.perf_counter()-start
        return result

    def generate(self,*args,**kwargs):
        self.times={'prediction_s':0.,'recovery_s':0.,'target_exposed_wait_s':0.,
                    'draft_forward_s':0.,'draft_queries':0,'draft_forward_calls':0}
        receive=self.target.receive
        forward=self.draft.forward
        def timed_receive():
            start=time.perf_counter()
            result=receive()
            self.times['target_exposed_wait_s']+=time.perf_counter()-start
            return result
        def timed_forward(seqs,tokens,**kw):
            torch.cuda.synchronize(1)
            start=time.perf_counter()
            result=forward(seqs,tokens,**kw)
            torch.cuda.synchronize(1)
            self.times['draft_forward_s']+=time.perf_counter()-start
            self.times['draft_queries']+=len(seqs)*len(tokens[0])
            self.times['draft_forward_calls']+=1
            return result
        self.target.receive=timed_receive
        self.draft.forward=timed_forward
        try:
            result=super().generate(*args,**kwargs)
            result['stage_profile']=dict(self.times)
            return result
        finally:
            self.target.receive=receive
            self.draft.forward=forward


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(TARGET)
    rows,cfg,_=original.load_rows('math500',32)
    prompts=[tok.encode(original._apply_chat_template(tok,[{'role':'user','content':cfg['question'](r)}],False)) for r in rows]
    data={'note':'Synchronized diagnosis, includes prefill. recovery includes bootstrap; target wait includes prefill/control IPC. draft_forward_s is nested within prediction/recovery: do not sum all fields.',
          'dataset':'math500','requests':32,'max_new_tokens':128,'concurrency':32,'cells':[]}
    e=ProfileEngine(TARGET,DRAFT)
    try:
        for anchor in [3,1]:
            e.A=anchor
            for policy in ['neural','fast_legacy','fast']:
                e.backup=policy
                e.generate(prompts,32,max_new_tokens=32)
                result=e.generate(prompts,32,max_new_tokens=128)
                result['anchor_num']=anchor
                data['cells'].append(result)
                a.output.write_text(json.dumps(data,indent=2))
                print(anchor,policy,result['tps'],result['stage_profile'],flush=True)
    finally:
        e.close()


if __name__=='__main__': main()
