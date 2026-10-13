import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "d3d_sglang_benchmark", ROOT / "scripts/sglang/benchmark.py"
)
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class SglangBenchmarkTests(unittest.TestCase):
    def test_acceptance_excludes_prefill_and_weights_by_verify_count(self):
        rows = [
            {"output_tokens": 17, "elapsed_seconds": 1.0,
             "meta_info": {"spec_verify_ct": 2}},
            {"output_tokens": 5, "elapsed_seconds": 2.0,
             "meta_info": {"spec_verify_ct": 2}},
        ]
        result = BENCHMARK.summarize(rows)
        self.assertEqual(result["acclen"], 5)
        self.assertEqual(result["accepted_draft_tokens_per_verify"], 4)
        self.assertAlmostEqual(result["tokens_per_second"], 22 / 3)

    def test_ar_has_no_fabricated_accept_length(self):
        result = BENCHMARK.summarize([
            {"output_tokens": 10, "elapsed_seconds": 1, "meta_info": {}}
        ])
        self.assertIsNone(result["acclen"])
        self.assertEqual(result["verify_calls"], 0)
