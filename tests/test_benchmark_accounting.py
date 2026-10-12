import json
import tempfile
import unittest
from pathlib import Path

from common.runtime import check_paged_capacity
from single.eval_sync import headline_metrics, summarize_timing_metrics


class BenchmarkAccountingTests(unittest.TestCase):
    def test_headline_metrics_report_speedup_and_accept_length(self):
        metrics = headline_metrics(d3d_tps=120.0, baseline_tps=40.0, accept_length=2.5)
        self.assertEqual(metrics['d3d_tps'], 120.0)
        self.assertEqual(metrics['baseline_tps'], 40.0)
        self.assertEqual(metrics['speedup'], 3.0)
        self.assertEqual(metrics['accept_length'], 2.5)
        self.assertIsNone(
            headline_metrics(d3d_tps=10.0, baseline_tps=0.0, accept_length=1.0)['speedup']
        )

    def test_decode_rate_excludes_prefill_first_token(self):
        rows = [{'output_tokens': 5, 'decode_tokens': 4, 'decode_elapsed_s': 2.0,
                 'tokens_per_s': 2.0, 'time_per_output_token': 0.5}]
        metrics = summarize_timing_metrics(rows, elapsed_s=3.0)
        self.assertEqual(metrics['aggregate_decode_tokens_per_s'], 2.0)
        self.assertEqual(metrics['elapsed_tokens_per_s'], 5 / 3)

    def test_paged_capacity_checks_context_and_branch_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'target'
            draft = Path(directory) / 'draft'
            for model_path in (target, draft):
                model_path.mkdir()
                (model_path / 'config.json').write_text(json.dumps({'max_position_embeddings': 4096}))
            prompts = {'example': [list(range(1000))]}
            with self.assertRaisesRegex(ValueError, 'pages'):
                check_paged_capacity(str(target), str(draft), prompts, [16], 2048, 10)
            with self.assertRaisesRegex(ValueError, 'context'):
                check_paged_capacity(str(target), str(draft), prompts, [1], 4096, 512)


if __name__ == '__main__':
    unittest.main()
