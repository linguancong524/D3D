"""Paged FlashAttention target; custom SDPA mask for D3D packed branches."""
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from ssd.utils.context import get_context

class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads, **kwargs):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.draft = kwargs.get("draft", False)
        self.num_kv_heads, self.scale = num_kv_heads, scale
        self.k_cache = self.v_cache = None

    def forward(self, q, k, v):
        c = get_context()
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        self.k_cache.view(-1, self.num_kv_heads, self.head_dim)[c.slot_mapping] = k
        self.v_cache.view(-1, self.num_kv_heads, self.head_dim)[c.slot_mapping] = v
        if c.is_prefill and self.draft:
            parts=[]
            bounds=c.cu_seqlens_q.cpu().tolist()
            for start,end in zip(bounds,bounds[1:]):
                qq=q[start:end].transpose(0,1).unsqueeze(0)
                kk=k[start:end].transpose(0,1).unsqueeze(0).repeat_interleave(self.num_heads//self.num_kv_heads,1)
                vv=v[start:end].transpose(0,1).unsqueeze(0).repeat_interleave(self.num_heads//self.num_kv_heads,1)
                mask=torch.ones(end-start,end-start,dtype=torch.bool,device=q.device).tril()
                oo=scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,scale=self.scale)
                parts.append(oo.squeeze(0).transpose(0,1))
            o=torch.cat(parts)
        elif c.is_prefill:
            o = flash_attn_varlen_func(q, k, v, c.cu_seqlens_q, c.cu_seqlens_q,
                                      c.max_seqlen_q, c.max_seqlen_q, causal=True,
                                      softmax_scale=self.scale)
        elif c.mode == 'target':
            o = flash_attn_with_kvcache(q.view(c.batch, c.qlen, self.num_heads, self.head_dim),
                    self.k_cache, self.v_cache, cache_seqlens=c.context_lens,
                    block_table=c.block_tables, causal=True, softmax_scale=self.scale)
        else:
            # Gather each request's logical prefix + transient packed queries.
            # Padding and branch isolation are encoded in one shared-head mask.
            keys = self.k_cache.view(-1, self.num_kv_heads, self.head_dim)[c.gather_slots]
            vals = self.v_cache.view(-1, self.num_kv_heads, self.head_dim)[c.gather_slots]
            keys = keys.transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, 1)
            vals = vals.transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, 1)
            queries = q.view(c.batch, c.qlen, self.num_heads, self.head_dim).transpose(1, 2)
            o = scaled_dot_product_attention(queries, keys, vals, attn_mask=c.mask,
                                            dropout_p=0, scale=self.scale)
            o = o.transpose(1, 2).contiguous()
        return o.reshape(-1, self.num_heads * self.head_dim)
