import unittest

import torch

from single.eval_async_packed import run_ar_baseline


class _FakeVerifier:
    """Minimal VerifierProcessClient stand-in that emits the next token id."""

    def __init__(self):
        self._next = None
        self.steps = 0

    def prefill(self, input_ids, temperature, target_top_k, target_top_p):
        self._next = 100
        return [self._next], 0.0, int(input_ids.shape[1])

    def submit_verify(self, block, draft_proposal, kv_len, temperature, target_top_k, target_top_p):
        self._next = int(block[0, 0]) + 1
        self.steps += 1
        return 0.0

    def receive_verify(self):
        return [self._next], 0, [self._next], {}


class ArBaselineTests(unittest.TestCase):
    def test_decode_tokens_exclude_the_prefill_token(self):
        client = _FakeVerifier()
        rows = run_ar_baseline(
            client,
            [torch.tensor([[1, 2, 3]])],
            max_new_tokens=5,
            eos_token_ids=[999],
            temperature=0.0,
            target_top_k=0,
            target_top_p=1.0,
        )
        self.assertEqual(rows[0]['output_tokens'], 5)
        self.assertEqual(rows[0]['decode_tokens'], 4)
        self.assertEqual(client.steps, 4)

    def test_eos_stops_the_baseline(self):
        client = _FakeVerifier()
        rows = run_ar_baseline(
            client,
            [torch.tensor([[1, 2, 3]])],
            max_new_tokens=8,
            eos_token_ids=[101],
            temperature=0.0,
            target_top_k=0,
            target_top_p=1.0,
        )
        self.assertEqual(rows[0]['output_tokens'], 2)
        self.assertEqual(rows[0]['decode_tokens'], 1)
        self.assertEqual(client.steps, 1)


if __name__ == '__main__':
    unittest.main()
