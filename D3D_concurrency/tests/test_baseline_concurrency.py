import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_concurrency import generate, slow_attempts


class FakeWorker:
    def __init__(self):
        self.active = set()
        self.pending = None
        self.max_active = 0

    def request(self, command):
        if command == 'reset_stats':
            assert not self.active
        return dict(active_sequences=len(self.active), free_pages=512, peak_allocated_gb=0)

    def prefill(self, seqs):
        self.active.update(s.seq_id for s in seqs)
        self.max_active = max(self.max_active, len(self.active))
        return [s.token_ids[0] for s in seqs]

    def submit_forward(self, seqs, blocks):
        assert all(s.prefix == len(s.token_ids) + len(s.generated) - 1 for s in seqs)
        self.pending = [[block[0] - 1] for block in blocks]

    def receive(self):
        return self.pending

    def release(self, seq):
        self.active.remove(seq.seq_id)


class BaselineTests(unittest.TestCase):
    def test_refill_eos_and_cap(self):
        worker = FakeWorker()
        result = generate(worker, [[0], [2], [8], [1], [3]], 2, 4, {0})
        self.assertEqual([item['tokens'] for item in result['outputs']],
                         [[0], [2, 1, 0], [8, 7, 6, 5], [1, 0], [3, 2, 1, 0]])
        self.assertEqual(result['output_tokens'], 14)
        self.assertEqual(result['length_limit_fraction'], 0.2)
        self.assertLessEqual(worker.max_active, 2)
        self.assertFalse(worker.active)

    def test_first_token_counts_and_no_decode_at_limit_one(self):
        worker = FakeWorker()
        result = generate(worker, [[3], [4], [5]], 16, 1, {0})
        self.assertEqual(result['output_tokens'], 3)
        self.assertEqual(result['decode_batch_sizes'], [])
        self.assertFalse(worker.active)

    def test_slow_detection(self):
        self.assertEqual(slow_attempts([dict(attempt=0, tps=100), dict(attempt=1, tps=84)], 0.85), [1])
        self.assertEqual(slow_attempts([dict(attempt=0, tps=100), dict(attempt=1, tps=85)], 0.85), [])
        cells = [dict(attempt=i, tps=tps) for i, tps in enumerate([100, 99, 70])]
        self.assertEqual(slow_attempts(cells, 0.85), [2])
        self.assertEqual(slow_attempts([dict(attempt=i, tps=tps) for i, tps in enumerate([100, 99, 98])], 0.85), [])


if __name__ == '__main__':
    unittest.main()
