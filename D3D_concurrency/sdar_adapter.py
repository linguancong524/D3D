"""Original SDAR model arithmetic with an SSD-style paged attention backend.

Only attention cache access changes: SDAR's norms, residuals, RoPE, MLP and
projection layers remain the exact implementation used by D3DV2.
"""
import types
import torch
from torch import nn
from ssd.layers.attention import Attention
from ssd.utils.context import get_context

def _attention(self,hidden_states,position_embeddings,attention_mask=None,**kwargs):
    from models.sdar.modeling_sdar import apply_rotary_pos_emb
    shape=hidden_states.shape[:-1]
    q=self.q_norm(self.q_proj(hidden_states).view(*shape,-1,self.head_dim)).transpose(1,2)
    k=self.k_norm(self.k_proj(hidden_states).view(*shape,-1,self.head_dim)).transpose(1,2)
    v=self.v_proj(hidden_states).view(*shape,-1,self.head_dim).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
    q=q.transpose(1,2).reshape(-1,self.attn.num_heads*self.head_dim)
    k=k.transpose(1,2).reshape(-1,self.attn.num_kv_heads*self.head_dim)
    v=v.transpose(1,2).reshape(-1,self.attn.num_kv_heads*self.head_dim)
    out=self.attn(q,k,v).view(*shape,-1)
    return self.o_proj(out),None

class SDARPageModel(nn.Module):
    def __init__(self,path,device):
        super().__init__()
        import eval_sync as original
        source=original._load_model(path,torch.bfloat16,attn_implementation='sdpa').to(device).eval()
        self.model=source.model
        self.lm_head=source.lm_head
        cfg=source.config
        for layer in self.model.layers:
            attn=layer.self_attn
            attn.attn=Attention(cfg.num_attention_heads,cfg.head_dim,cfg.head_dim**-0.5,
                                cfg.num_key_value_heads,draft=True)
            attn.forward=types.MethodType(_attention,attn)

    def forward(self,ids,positions):
        c=get_context()
        B=1 if c.is_prefill else c.batch
        return self.model(input_ids=ids.view(B,-1),position_ids=positions.view(B,-1),
                          use_cache=False,return_dict=True).last_hidden_state.reshape(-1,self.lm_head.in_features)
