import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


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

    def test_concurrent_throughput_uses_wall_time(self):
        rows = [
            {"output_tokens": 10, "elapsed_seconds": 2, "meta_info": {}}
            for index in range(4)
        ]
        result = BENCHMARK.summarize(rows, elapsed_seconds=2)
        self.assertEqual(result["tokens_per_second"], 20)
        self.assertEqual(result["elapsed_seconds"], 2)

    def test_full_dataset_keeps_every_prompt_and_uses_separate_warmup(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: [messages[0]["content"]]
        transformers = SimpleNamespace(AutoTokenizer=Mock())
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        generate = Mock(return_value={"text": "answer", "meta_info": {"completion_tokens": 3}})
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "code.jsonl"
            dataset.write_text('{"prompt": "Write a Python function."}\n{"text": "Explain a loop."}\n')
            output = Path(directory) / "full.json"
            arguments = ["benchmark.py", "--dataset", str(dataset), "--tokenizer", "mock",
                         "--samples", "0", "--repetitions", "1", "--output", str(output)]
            with patch.dict(sys.modules, {"transformers": transformers}), \
                    patch.object(sys, "argv", arguments), \
                    patch.object(BENCHMARK, "call", generate), patch("builtins.print"):
                BENCHMARK.main()
            report = json.loads(output.read_text())
            self.assertEqual(report["indices"], [0, 1])
            self.assertIsNone(report["warmup_index"])
            self.assertEqual([row["input_ids"] for row in report["rows"]],
                             [["Write a Python function."], ["Explain a loop."]])
            self.assertNotIn(report["warmup_input_ids"], [row["input_ids"] for row in report["rows"]])
            self.assertEqual(len(output.with_suffix(".rows.jsonl").read_text().splitlines()), 2)

    def test_cli_sends_concurrent_requests_and_keeps_prompt_order(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        def generate(base_url, endpoint, payload):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return {"text": str(payload["input_ids"]),
                    "meta_info": {"completion_tokens": 10}}

        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: [int(messages[0]["content"].splitlines()[0])]
        )
        transformers = SimpleNamespace(AutoTokenizer=Mock())
        transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            dataset.write_text("\n".join(
                json.dumps({"problem": str(index)}) for index in range(9)
            ))
            for concurrency in (1, 4):
                peak = 0
                output = Path(directory) / f"c{concurrency}.json"
                arguments = [
                    "benchmark.py", "--dataset", str(dataset),
                    "--tokenizer", "mock", "--samples", "8",
                    "--concurrency", str(concurrency), "--repetitions", "1",
                    "--output", str(output),
                ]
                with patch.dict(sys.modules, {"transformers": transformers}), \
                        patch.object(sys, "argv", arguments), \
                        patch.object(BENCHMARK, "call", side_effect=generate), \
                        patch("builtins.print"):
                    BENCHMARK.main()
                report = json.loads(output.read_text())
                self.assertEqual(peak, concurrency)
                self.assertEqual(report["summary"]["requests"], 8)
                self.assertEqual(
                    [row["source_index"] for row in report["rows"]], report["indices"]
                )
                if concurrency > 1:
                    self.assertLess(
                        report["summary"]["elapsed_seconds"],
                        sum(row["elapsed_seconds"] for row in report["rows"]),
                    )
