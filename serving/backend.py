"""SSD model layers and page allocator with D3D invocation semantics."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen3Config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from ssd.models.qwen3 import Qwen3ForCausalLM
from ssd.layers.rotary_embedding import get_rope
from ssd.engine.block_manager import BlockManager
from ssd.engine.sequence import Sequence
from ssd.sampling_params import SamplingParams  # noqa: F401 (re-exported for serving.engine)
from ssd.utils.loader import load_model
from ssd.utils.context import set_context, reset_context
from serving.d3d_adapter import D3DPageModel

PAGE = 256
MASK_ID = 151669

class Runner:
    def __init__(self, path, device, *, draft=False, pages=512):
        self.device = torch.device(device)
        self.draft = draft
        cfg = Qwen3Config(**json.loads((Path(path) / 'config.json').read_text()))
        cfg.rope_scaling = None
        cfg.rope_theta = json.loads((Path(path) / 'config.json').read_text()).get('rope_theta', 1000000)
        cfg.max_position_embeddings = min(cfg.max_position_embeddings, 16384)
        self.config = cfg
        get_rope.cache_clear()
        self.pool = BlockManager(pages, PAGE, is_draft=draft, max_model_len=16384)
        Sequence.block_size = PAGE
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(self.device):
            if draft:
                self.model = D3DPageModel(path,self.device).eval()
            else:
                self.model = Qwen3ForCausalLM(cfg).eval()
                load_model(self.model, str(path))
            for layer in self.model.model.layers:
                a = layer.self_attn.attn
                a.k_cache = torch.zeros(pages, PAGE, cfg.num_key_value_heads, cfg.head_dim,
                                        dtype=torch.bfloat16, device=self.device)
                a.v_cache = torch.zeros_like(a.k_cache)
        torch.set_default_dtype(old)
        torch.cuda.synchronize(self.device)

    def table(self, seq):
        return seq.draft_block_table if self.draft else seq.block_table

    def reserve(self, seq, length):
        if length > self.config.max_position_embeddings:
            raise ValueError(f'Context length {length} exceeds {self.config.max_position_embeddings}')
        table = self.table(seq)
        missing = (length + PAGE - 1) // PAGE - len(table)
        if missing > 0:
            table.extend(b.block_id for b in self.pool._allocate_n_blocks(missing))

    def release(self, seq):
        self.pool.deallocate(seq)

    def trim(self, seq, length):
        table = self.table(seq)
        keep = (length + PAGE - 1) // PAGE
        if len(table) > keep:
            self.pool._deallocate_n_blocks(table[keep:])
            del table[keep:]

    def _tables(self, seqs):
        width = max(len(self.table(s)) for s in seqs)
        return torch.tensor([self.table(s) + [0] * (width-len(self.table(s))) for s in seqs],
                            dtype=torch.int32, device=self.device)

    def _run(self, ids, pos, context, *, output_topk=0, selected=None):
        token = set_context(context)
        try:
            hidden = self.model(ids.reshape(-1), pos.reshape(-1))
            if selected is not None:
                hidden = hidden[selected]
            # Bound full-vocabulary projection memory, particularly B x branches x block.
            parts = []; greedy_parts = []
            for chunk in hidden.split(256):
                logits = torch.nn.functional.linear(chunk, self.model.lm_head.weight)
                greedy = logits.argmax(-1)
                if output_topk:
                    parts.append(logits.topk(output_topk, dim=-1).indices)
                    greedy_parts.append(greedy)
                else: parts.append(greedy)
            return (torch.cat(parts), torch.cat(greedy_parts)) if output_topk else torch.cat(parts)
        finally:
            reset_context(token)

    @torch.inference_mode()
    def prefill(self, seqs):
        torch.cuda.set_device(self.device)
        lengths = [len(s.token_ids) for s in seqs]
        ids, positions, slots = [], [], []
        for s, n in zip(seqs, lengths):
            self.reserve(s, n)
            ids.extend(s.token_ids); positions.extend(range(n))
            bt = self.table(s)
            slots.extend(bt[i // PAGE] * PAGE + i % PAGE for i in range(n))
        cu = torch.tensor([0] + lengths, dtype=torch.int32, device=self.device).cumsum(0, dtype=torch.int32)
        c = SimpleNamespace(is_prefill=True, mode='prefill', cu_seqlens_q=cu,
                            max_seqlen_q=max(lengths), slot_mapping=torch.tensor(slots, device=self.device))
        out = self._run(torch.tensor(ids, device=self.device), torch.tensor(positions, device=self.device),
                        c, selected=cu[1:].long()-1)
        return out.cpu().tolist()

    @torch.inference_mode()
    def forward(self, seqs, tokens, *, positions=None, mask_builder=None, topk=0):
        torch.cuda.set_device(self.device)
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        B, Q = ids.shape
        prefixes = [s.prefix for s in seqs]
        for s, n in zip(seqs, prefixes): self.reserve(s, n+Q)
        bt = self._tables(seqs)
        starts = torch.tensor(prefixes, device=self.device)
        logical = starts[:, None] + torch.arange(Q, device=self.device)[None, :]
        slots = bt.long().gather(1, logical // PAGE) * PAGE + logical % PAGE
        pos = logical if positions is None else torch.as_tensor(positions, device=self.device)
        c = SimpleNamespace(is_prefill=False, mode='draft' if self.draft else 'target',
                            batch=B, qlen=Q, slot_mapping=slots.reshape(-1), block_tables=bt,
                            context_lens=(starts+Q).int(), cu_seqlens_q=None)
        if self.draft:
            S = max(prefixes) + Q
            logical_keys = torch.arange(S, device=self.device)[None, :].expand(B, -1)
            page_indices = (logical_keys // PAGE).clamp(max=bt.shape[1]-1)
            c.gather_slots = bt.long().gather(1, page_indices)*PAGE + logical_keys % PAGE
            c.mask = mask_builder(starts, Q, S)
        result = self._run(ids, pos, c, output_topk=topk)
        return (result[0].view(B,Q,topk), result[1].view(B,Q)) if topk else result.view(B,Q)
