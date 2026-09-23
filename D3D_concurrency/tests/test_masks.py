"""Attention visibility is the essential SDAR semantic contract."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from engine import Engine

class Masks(unittest.TestCase):
    def setUp(self):
        self.e=Engine.__new__(Engine);self.e.K=2

    def test_packed_visibility_variable_prefix(self):
        specs=[[(1,7),(2,None)],[(1,9),(2,None)]]
        prefixes=torch.tensor([2,4])
        mask=self.e._mask(specs)(prefixes,6,10)[:,0]
        for b,p in enumerate(prefixes.tolist()):
            for q in range(6):
                visible=mask[b,q].nonzero().flatten().tolist()
                if q<2: expected=list(range(p+q+1))
                elif q<4: expected=list(range(p+1))+list(range(p+2,p+4))
                else: expected=list(range(p+2))+list(range(p+4,p+6))
                self.assertEqual(visible,expected)

    def test_recovery_full_block_without_padding(self):
        mask=self.e._mask()(torch.tensor([1,3]),2,5)[:,0]
        self.assertTrue(mask[0,:,:3].all());self.assertFalse(mask[0,:,3:].any())
        self.assertTrue(mask[1].all())

class PageOwnership(unittest.TestCase):
    def test_rollback_release_and_reuse(self):
        from backend import Runner, Sequence, SamplingParams
        from ssd.engine.block_manager import BlockManager
        Sequence.block_size=256
        runner=Runner.__new__(Runner);runner.draft=False
        runner.pool=BlockManager(8,256)
        first=Sequence([1]*300,SamplingParams())
        second=Sequence([2]*17,SamplingParams())
        runner.reserve(first,1100);runner.reserve(second,600)
        self.assertFalse(set(first.block_table)&set(second.block_table))
        original=first.block_table[:]
        runner.trim(first,301)
        self.assertEqual(first.block_table,original[:2])
        runner.release(first);runner.release(second)
        self.assertEqual(len(runner.pool.free_block_ids),8)
        self.assertFalse(runner.pool.used_block_ids)

class GreedyTies(unittest.TestCase):
    def test_argmax_is_not_taken_from_topk(self):
        from backend import Runner
        from types import SimpleNamespace
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head=SimpleNamespace(weight=torch.eye(4))
            def forward(self,ids,positions):
                return torch.tensor([[3.,3.,0.,-1.]])
        runner=Runner.__new__(Runner);runner.model=Fixed()
        top,greedy=runner._run(torch.tensor([0]),torch.tensor([0]),SimpleNamespace(),output_topk=2)
        self.assertEqual(greedy.item(),0)
        self.assertEqual(set(top[0].tolist()),{0,1})

if __name__=='__main__':unittest.main()
