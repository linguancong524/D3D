"""Strict-concurrency D3D greedy engine; one target and one D3D instance.

SSD Sequence/page ownership reused. Request admission is capped explicitly.
Target verification executes in a dedicated process on GPU0 while GPU1 predicts
all selected acceptance/anchor outcomes with one D3D forward per microbatch.
"""
from __future__ import annotations
from serving.target_process import TargetProcess
import time
import torch
from serving.backend import Runner, Sequence, SamplingParams, MASK_ID
from common.runtime import check_hardware, resolve_model_path

class Engine:
    def __init__(self, target, draft, block_size=16, anchor_num=3, draft_microbatch=4, pages=512, backup="neural", seed=42, fast_miss_anchors=0, draft_pages=None):
        check_hardware()
        target = resolve_model_path(target)
        draft = resolve_model_path(draft)
        if backup not in ("neural", "fast", "fast_legacy"): raise ValueError(backup)
        if anchor_num < 1: raise ValueError('anchor_num must be positive')
        if fast_miss_anchors < 0: raise ValueError('fast_miss_anchors must be nonnegative')
        self.fast_miss_anchors=fast_miss_anchors
        self.backup, self.seed = backup, seed
        self.K, self.A, self.micro = block_size, anchor_num, draft_microbatch
        self.target = TargetProcess(target, pages)
        try:
            self.draft = Runner(draft, 'cuda:1', draft=True, pages=pages if draft_pages is None else draft_pages)
            self.target.receive()
        except BaseException:
            self.target.close()
            raise

    def close(self): self.target.close()

    def _mask(self, specs=None):
        def build(starts, Q, S):
            dev = starts.device
            cols = torch.arange(S, device=dev)[None,None,:].expand(len(starts),-1,-1)
            pref = starts[:,None,None]
            local = cols-pref
            if specs is None: # recovery: all block tokens attend each other
                mask = (cols < pref+Q).expand(-1,Q,-1).clone()
            else:
                mask = torch.zeros(len(starts),Q,S,dtype=torch.bool,device=dev)
                rows = torch.arange(self.K,device=dev)[None,:,None]
                mask[:,:self.K] = (cols < pref) | ((local>=0)&(local<=rows))
                for b, branches in enumerate(specs):
                    for j,(accepted,anchor) in enumerate(branches):
                        off=self.K+j*self.K
                        mask[b,off:off+self.K] = (cols[b] < pref[b]+accepted) | ((local[b]>=off)&(local[b]<off+self.K))
            return mask[:,None]
        return build

    def _recover(self, seqs):
        for off in range(0,len(seqs),self.micro):
            group=seqs[off:off+self.micro]
            ids=torch.full((len(group),self.K),MASK_ID,dtype=torch.long,device=self.draft.device)
            ids[:,0]=torch.tensor([s.anchor for s in group],device=ids.device)
            top,blocks=self.draft.forward(group,ids,mask_builder=self._mask(),topk=self.A+1)
            blocks[:,0]=ids[:,0]
            for s,block,t in zip(group,blocks,top):
                s.current=block; s.top=t; s.recoveries+=1

    def _fast_backup(self, seqs):
        # Greedy verification needs token IDs only, not random full-vocab logits.
        for s in seqs:
            s.current=torch.randint(self.draft.config.vocab_size,(self.K,),
                                    generator=self.backup_rng,device=self.draft.device)
            s.current[0]=s.anchor
            s.needs_fast_seed=True
            s.fast_backups+=1

    def _predict(self,seqs):
        # Target is already verifying the random blocks. D3D needs masked
        # predictions to choose useful anchor branches (unmasked-token logits
        # are not AR next-token logits). Build these now, off the pre-submit path.
        # Keep random current blocks unchanged: these are the target's inputs.
        missing=[s for s in seqs if getattr(s,'needs_fast_seed',False)]
        missing_ids={s.seq_id for s in missing}
        miss_anchors=(getattr(self,'fast_miss_anchors',0) or self.A) if self.backup=='fast' else self.A
        for off in range(0,len(missing),self.micro):
            group=missing[off:off+self.micro]
            ids=torch.full((len(group),self.K),MASK_ID,dtype=torch.long,device=self.draft.device)
            ids[:,0]=torch.tensor([s.anchor for s in group],device=ids.device)
            top,_=self.draft.forward(group,ids,mask_builder=self._mask(),topk=max(self.A,miss_anchors)+1)
            for s,t in zip(group,top):
                s.top=t; s.needs_fast_seed=False; s.fast_seeds+=1
        results={}
        # Different query lengths need separate microbatches. A random miss
        # almost always accepts only its verified anchor; deeper outcomes can
        # safely miss the prediction cache and use backup again.
        if self.backup == 'fast':
            partitions=[([s for s in seqs if s.seq_id not in missing_ids], False),
                        (missing, True)]
        else:
            partitions=[(seqs, False)]
        groups=[(part[off:off+self.micro], shallow)
                for part,shallow in partitions for off in range(0,len(part),self.micro)]
        for group,shallow in groups:
            specs=[]
            for s in group:
                top=s.top.cpu().tolist(); current=s.current.cpu().tolist()
                branches=[]
                fanout=miss_anchors if shallow else self.A
                for accepted in (range(1,2) if shallow else range(1,self.K)):
                    anchors=[v for v in top[accepted] if v != current[accepted]][:fanout]
                    assert len(anchors)==fanout
                    branches.extend((accepted,a) for a in anchors)
                if not shallow: branches.append((self.K,None))
                specs.append(branches)
            Q=self.K+len(specs[0])*self.K
            ids=torch.full((len(group),Q),MASK_ID,dtype=torch.long,device=self.draft.device)
            pos=torch.empty_like(ids)
            for b,s in enumerate(group):
                ids[b,:self.K]=s.current
                pos[b,:self.K]=torch.arange(s.prefix,s.prefix+self.K,device=ids.device)
                for j,(n,a) in enumerate(specs[b]):
                    start=self.K+j*self.K
                    if a is not None: ids[b,start]=a
                    pos[b,start:start+self.K]=torch.arange(s.prefix+n,s.prefix+n+self.K,device=ids.device)
            top,greedy=self.draft.forward(group,ids,positions=pos,mask_builder=self._mask(specs),topk=self.A+1)
            for b,s in enumerate(group):
                outcomes={}
                for j,key in enumerate(specs[b]):
                    start=self.K+j*self.K
                    t=top[b,start:start+self.K].clone()
                    block=greedy[b,start:start+self.K].clone()
                    if key[1] is not None: block[0]=key[1]
                    outcomes[key]=(block,t)
                results[s.seq_id]=outcomes
        return results

    @torch.inference_mode()
    def _verify(self,seqs,blocks):
        self.target.submit_forward(seqs,blocks)
        return self._accepted(blocks,self.target.receive())

    @staticmethod
    def _accepted(blocks,posterior):
        out=[]
        for block,p in zip(blocks,posterior):
            accepted=1
            while accepted<len(block) and block[accepted]==p[accepted-1]: accepted+=1
            out.append((accepted,p[accepted-1]))
        return out

    @torch.inference_mode()
    def generate(self,prompts,concurrency,max_new_tokens=256,ignore_eos=False,eos=(151643,151645)):
        if concurrency<1 or max_new_tokens<1: raise ValueError('positive concurrency and output length required')
        waiting=iter(enumerate(prompts)); running=[]; outputs=[]; exhausted=False
        self.backup_rng=torch.Generator(device=self.draft.device).manual_seed(self.seed)
        started=time.perf_counter(); max_active=0; batch_sizes=[]
        miss_rounds=0; miss_requests=0
        self.target.request('reset_stats')
        torch.cuda.reset_peak_memory_stats(1)
        while running or not exhausted:
            admitted=[]
            while len(running)+len(admitted)<concurrency and not exhausted:
                try: idx,ids=next(waiting)
                except StopIteration: exhausted=True; break
                s=Sequence(ids,SamplingParams(temperature=0,max_new_tokens=max_new_tokens))
                s.index=idx; s.prefix=len(ids); s.generated=[]; s.born=time.perf_counter()
                s.accepts=[]; s.hits=0; s.recoveries=0
                s.fast_backups=0; s.fast_seeds=0; s.needs_fast_seed=False
                admitted.append(s)
            if admitted:
                # Prefill runs concurrently on target/draft and in bounded token batches.
                for off in range(0,len(admitted),4):
                    group=admitted[off:off+4]
                    self.target.submit_prefill(group)
                    self.draft.prefill(group)
                    anchors=self.target.receive()
                    ready=[]
                    for s,a in zip(group,anchors):
                        s.anchor=a
                        if not ignore_eos and a in eos:
                            s.generated=[a]; self._finish(s,outputs)
                        else: ready.append(s)
                    self._recover(ready)
                    running.extend(ready)
            if not running: continue
            max_active=max(max_active,len(running)); batch_sizes.append(len(running))
            blocks=[s.current.cpu().tolist() for s in running]
            self.target.submit_forward(running,blocks)
            # Packed forward also commits actual pending tokens to draft KV.
            predictions=self._predict(running)
            accepted=self._accepted(blocks,self.target.receive())
            remaining=[]; recover=[]
            for s,block,(n,anchor) in zip(running,blocks,accepted):
                emitted=block[:n]
                room=max_new_tokens-len(s.generated)
                emitted=emitted[:room]
                stop=False
                if not ignore_eos:
                    for j,t in enumerate(emitted):
                        if t in eos: emitted=emitted[:j+1]; stop=True; break
                s.generated.extend(emitted); s.accepts.append(n)
                s.prefix+=n; s.anchor=anchor
                if stop or len(s.generated)>=max_new_tokens:
                    self._finish(s,outputs); continue
                # Anchor is target-produced; stop immediately if it is EOS.
                if not ignore_eos and anchor in eos:
                    s.generated.append(anchor); self._finish(s,outputs); continue
                key=(n,None if n==self.K else anchor)
                selected=predictions[s.seq_id].get(key)
                if selected is None: recover.append(s)
                else:
                    s.current,s.top=selected; s.current[0]=anchor; s.hits+=1
                self.target.trim(s,s.prefix); self.draft.trim(s,s.prefix)
                remaining.append(s)
            running=remaining
            miss_rounds+=bool(recover); miss_requests+=len(recover)
            if self.backup in ('fast','fast_legacy'): self._fast_backup(recover)
            else: self._recover(recover)
        target_stats=self.target.request('stats')
        torch.cuda.synchronize(1)
        assert target_stats['active_sequences']==0, 'Target sequences leaked'
        assert len(self.draft.pool.free_block_ids)==len(self.draft.pool.blocks), 'Draft pages leaked'
        elapsed=time.perf_counter()-started
        outputs.sort(key=lambda s:s['index'])
        total=sum(len(s['tokens']) for s in outputs)
        return dict(outputs=outputs,elapsed_s=elapsed,output_tokens=total,tps=total/elapsed,
                    max_active=max_active,decode_batch_sizes=batch_sizes,
                    backup=self.backup,miss_rounds=miss_rounds,miss_requests=miss_requests,
                    peak_allocated_gb=[target_stats['peak_allocated_gb'],torch.cuda.max_memory_allocated(1)/1e9],
                    target_active_sequences=target_stats['active_sequences'],
                    target_free_pages=target_stats['free_pages'])

    def _finish(self,s,outputs):
        outputs.append(dict(index=s.index,tokens=s.generated,latency_s=time.perf_counter()-s.born,
                            accepts=s.accepts,hits=s.hits,recoveries=s.recoveries,
                            fast_backups=s.fast_backups,fast_seeds=s.fast_seeds))
        self.target.release(s); self.draft.release(s)
