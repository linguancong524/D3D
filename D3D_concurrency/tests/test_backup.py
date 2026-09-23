"""Fast misses must keep a verified anchor, and cannot commit unchecked suffixes."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from engine import Engine

class Backup(unittest.TestCase):
    def test_mixed_fast_prediction_keeps_request_keys_and_sparse_misses(self):
        e=Engine.__new__(Engine)
        e.K=4; e.A=2; e.micro=4; e.backup='fast'
        calls=[]
        def forward(group,ids,**kwargs):
            calls.append((tuple(s.seq_id for s in group),ids.shape[1]))
            top=torch.arange(1,kwargs['topk']+1).expand(len(group),ids.shape[1],kwargs['topk']).clone()
            return top,torch.ones_like(ids)
        e.draft=SimpleNamespace(device=torch.device('cpu'),forward=forward)
        seqs=[SimpleNamespace(seq_id=i,prefix=7,anchor=5,current=torch.tensor([5,6,6,6]),
                              top=torch.tensor([1,2,3]).expand(4,3),needs_fast_seed=i==1,
                              fast_seeds=0) for i in range(2)]
        results=e._predict(seqs)
        self.assertEqual(calls,[((1,),4),((0,),32),((1,),12)])
        self.assertEqual(set(results[1]),{(1,1),(1,2)})
        self.assertIn((4,None),results[0])
        self.assertNotIn((2,1),results[1])
        self.assertEqual(seqs[1].current.tolist(),[5,6,6,6])
        self.assertFalse(seqs[1].needs_fast_seed)
        for key,(block,_) in results[1].items():
            self.assertEqual(block[0].item(),key[1])
        calls.clear()
        e.fast_miss_anchors=5
        seqs[1].needs_fast_seed=True
        wider=e._predict(seqs)
        self.assertEqual(calls,[((1,),4),((0,),32),((1,),24)])
        self.assertEqual(set(wider[1]),{(1,a) for a in range(1,6)})

    def test_fast_does_not_call_neural_before_target_submission(self):
        e=Engine.__new__(Engine);e.K=16
        e.draft=SimpleNamespace(device=torch.device('cpu'),config=SimpleNamespace(vocab_size=101))
        e.backup_rng=torch.Generator().manual_seed(42)
        seqs=[SimpleNamespace(anchor=a,fast_backups=0) for a in [3,100]]
        e._fast_backup(seqs)
        for s in seqs:
            self.assertEqual(s.current[0].item(),s.anchor)
            self.assertTrue(((s.current>=0)&(s.current<101)).all())
            self.assertTrue(s.needs_fast_seed)
            self.assertEqual(s.fast_backups,1)
            self.assertEqual(len(s.current),16)
        # Random suffix mismatch emits just the already verified anchor.
        self.assertEqual(e._accepted([[3,90,91]],[[7,8,9]]),[(1,7)])
        # Even accidentally matching random tokens must be checked in order.
        self.assertEqual(e._accepted([[3,7,91]],[[7,8,9]]),[(2,8)])

if __name__=='__main__':unittest.main()
