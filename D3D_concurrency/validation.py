"""GPU regression checks against original D3DV2 SDAR operations."""
import torch
from transformers.cache_utils import DynamicCache
import eval_sync as original
import eval_async_packed as async_original
from backend import Sequence,SamplingParams

@torch.inference_mode()
def validate_draft(engine,path,prompt):
    ref=original._load_model(path,torch.bfloat16,attn_implementation='sdpa').to('cuda:1').eval()
    original._disable_default_cache(ref)
    seq=Sequence(prompt,SamplingParams(temperature=0))
    seq.prefix=len(prompt);seq.recoveries=0
    seq.anchor=engine.target.prefill([seq])[0]
    engine.draft.prefill([seq])
    engine._recover([seq])
    ids=torch.tensor([prompt],device='cuda:1')
    kv=DynamicCache()
    ref(ids,position_ids=original._pos(0,len(prompt),ids.device),
        attention_mask=original._prefix_attention_mask(len(prompt),ids.device,'causal',16),
        past_key_values=kv,use_cache=True,store_kv=True)
    anchor=torch.tensor([seq.anchor],device=ids.device)
    block,top,_=async_original.sdar_draft_from_anchor(ref,anchor=anchor,block_size=16,
            prefix_len=len(prompt),draft_kv=kv,dtype=ids.dtype,denoising_steps=1,
            threshold=0.9,remasking='low_confidence',temperature=0,top_k=64,top_p=1,
            mask_id=original.MASK_ID,attention_mask_mode='causal',return_topk=4)
    recovery_equal=block[0].tolist()==seq.current.tolist()
    # Use identical pending input/top-k to isolate packed forward correctness.
    seq.current=block[0].clone();seq.top=top[0].clone()
    ours=engine._predict([seq])[seq.seq_id]
    expected,_,_=async_original.sdar_draft_all_outcomes_packed(ref,pending_block=block,
            branch_token_count=16,draft_kv=kv,draft_kv_len=len(prompt),temperature=0,
            top_k=64,top_p=1,mask_id=original.MASK_ID,attention_mask_mode='causal',
            pending_top_tokens=top,anchor_num=3,accepted_counts=None)
    assert set(ours)==set(expected),'Packed outcome key mismatch'
    equal=total=0
    differences=[]
    for key in expected:
        x=ours[key][0].tolist();y=expected[key][0].tolist()
        equal+=sum(a==b for a,b in zip(x,y));total+=len(x)
        if x!=y:differences.append({'key':key,'actual':x,'expected':y})
    result={'recovery_equal':recovery_equal,'candidate_keys':len(expected),
            'packed_tokens_equal':equal,'packed_tokens_total':total,'differences':differences}
    engine.target.release(seq);engine.draft.release(seq)
    del ref;torch.cuda.empty_cache()
    print('SDAR VALIDATION', {k:v for k,v in result.items() if k!='differences'},flush=True)
    # BF16 implementation differences may flip close logits; report exact agreement.
    import json
    from pathlib import Path
    (Path(__file__).parent/'results/draft_validation.json').write_text(json.dumps(result,indent=2))
    if not recovery_equal or equal/total<0.99:
        raise RuntimeError('SDAR forward differs materially from D3DV2 reference')
    return result
