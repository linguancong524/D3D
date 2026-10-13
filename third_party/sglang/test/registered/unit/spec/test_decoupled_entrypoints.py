"""Configuration gates and aggregate benchmark correctness on CPU."""

# pyright: reportMissingImports=false

import asyncio
import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from sglang.benchmark.decoupled_serving import (
    Result,
    compare_outputs,
    encode_dataset_prompts,
    run,
    summarize,
)
from sglang.srt.speculative.decoupled.validation import (
    configure_target,
    validate_request,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def target_args(**changes):
    values = dict(
        device="cuda",
        skip_tokenizer_init=False,
        disaggregation_mode="null",
        decoupled_draft_endpoint="tcp://127.0.0.1:32000",
        decoupled_draft_timeout=30,
        decoupled_lookup_wait=0,
        decoupled_wait_for_covered_result=False,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        enable_lora=False,
        enable_two_batch_overlap=False,
        speculative_draft_model_path="local-sdar",
        # Regression: Draft anchor top-k must not leak into Target tree top-k.
        speculative_eagle_topk=4,
        decoupled_spec_role="null",
        speculative_num_draft_tokens=None,
        max_running_requests=None,
    )
    return SimpleNamespace(**(values | changes))


class TestDecoupledConfiguration(CustomTestCase):
    def test_host_candidate_contract_and_configuration(self):
        """Draft anchor top-k must not turn linear Target verify into a tree."""
        algo = SpeculativeAlgorithm.from_string("DECOUPLED")
        self.assertFalse(algo.is_host_draft())
        self.assertFalse(algo.is_ngram())
        self.assertTrue(algo.has_draft_kv())
        args = target_args()
        configure_target(args)
        self.assertEqual(args.speculative_num_draft_tokens, 16)
        self.assertEqual(args.speculative_num_steps, 15)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertTrue(args.disable_overlap_schedule)

    def test_unsupported_combinations_fail_before_model_startup(self):
        for changes in (
            {"tp_size": 2},
            {"dp_size": 3},
            {"pp_size": 2},
            {"enable_lora": True},
            {"enable_dp_attention": True},
            {"enable_two_batch_overlap": True},
            {"decoupled_spec_role": "verify"},
            {"decoupled_draft_timeout": 0},
            {"decoupled_wait_for_covered_result": True},
            {"speculative_num_draft_tokens": 1},
            {"speculative_draft_model_path": None},
            {"skip_tokenizer_init": True},
            {"device": "cpu"},
            {"disaggregation_mode": "decode"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                configure_target(target_args(**changes))

    def test_sampling_and_multimodal_rejection(self):
        req = SimpleNamespace(
            sampling_params=SimpleNamespace(top_k=1), multimodal_inputs=None
        )
        self.assertIsNone(validate_request(req))
        req.sampling_params.top_k = 20
        self.assertIsNotNone(validate_request(req))
        req.sampling_params.top_k = 1
        req.multimodal_inputs = object()
        self.assertIsNotNone(validate_request(req))

    def test_remote_branch_group_geometry_matches_candidate_batching(self):
        root = Path(__file__).resolve().parents[4]
        spec = importlib.util.spec_from_file_location(
            "decoupled_launch_geometry", root / "scripts/decoupled/launch.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        geometry = SimpleNamespace(
            branch_group_size=3,
            packed_single_row=False,
            num_anchors=3,
            accept_counts=list(range(1, 9)),
            block_size=8,
            max_branches=22,
            speculation_batch_size=8,
        )
        self.assertEqual(module.remote_packed_tokens_per_item(geometry), 240)
        self.assertEqual(module.remote_packed_graph_max_tokens(geometry), 1920)
        self.assertEqual(
            module.remote_branch_group_count(
                SimpleNamespace(
                    max_branches=29,
                    branch_group_size=3,
                    packed_single_row=False,
                    num_anchors=4,
                    accept_counts=list(range(1, 9)),
                    block_size=8,
                )
            ),
            15,
        )
        self.assertEqual(
            module.remote_branch_group_count(
                SimpleNamespace(
                    max_branches=22,
                    branch_group_size=1,
                    packed_single_row=False,
                    num_anchors=3,
                    accept_counts=list(range(1, 9)),
                    block_size=8,
                )
            ),
            22,
        )
        geometry.packed_single_row = True
        self.assertEqual(module.remote_branch_group_count(geometry), 1)
        self.assertEqual(module.remote_packed_tokens_per_item(geometry), 184)
        self.assertEqual(module.remote_packed_graph_max_tokens(geometry), 1472)

    def test_four_gpu_launch_preserves_ar_overlap_and_shared_draft(self):
        root = Path(__file__).resolve().parents[4]
        spec = importlib.util.spec_from_file_location(
            "decoupled_launch", root / "scripts/decoupled/launch.py"
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(
            mode="ar4",
            num_targets=3,
            gpus=[0, 1, 2, 3],
            target_model="target",
            target_mem_fraction_static=0.68,
            target_chunked_prefill_size=2048,
            disable_prefill_cuda_graph=True,
            draft_model="draft",
            remote_draft_model=None,
            draft_backend="engine",
            mask_token_id=31,
            block_size=16,
            draft_port=32000,
            draft_endpoint=None,
            base_port=30000,
            num_anchors=0,
            max_branches=16,
            context_length=8192,
            max_running_requests=48,
            draft_model_class="sdar:Model",
            draft_tokenizer=None,
            accept_counts=None,
            trust_remote_code=False,
            draft_timeout=30,
            lookup_wait_ms=0,
            wait_for_covered_result=False,
            candidate_topk=4,
            branch_group_size=3,
            packed_single_row=False,
            batched_branch_topk=False,
            packed_causal_pending=False,
            engine_deterministic=False,
            poll_interval_ms=1,
            shadow_local=False,
            draft_batch_size=64,
            speculation_batch_size=8,
            batching_window_ms=1,
            draft_metrics_interval=10,
            draft_prefill_chunk_size=1024,
            draft_attention_backend="flashinfer",
            draft_dtype="bfloat16",
            remote_draft_quantization=None,
            draft_mem_fraction_static=0.85,
            draft_enable_cuda_graph=False,
            draft_cuda_graph_max_tokens=None,
            local_prefix_reuse=False,
            local_cuda_graph_max_tokens=0,
            local_cuda_graph_max_requests=1,
            local_cuda_graph_check=False,
            timing_sync=False,
            dllm_algorithm="OneStep",
            dllm_algorithm_config="sdar.yaml",
        )
        args.draft_cuda_graph_max_tokens = 1024
        ar = module.commands(args)
        self.assertEqual([gpu for _, gpu, _ in ar], [0, 1, 2, 3])
        self.assertTrue(
            all("--disable-overlap-schedule" not in cmd for _, _, cmd in ar)
        )
        args.mode = "hybrid"
        packed = module.commands(args)
        self.assertEqual(
            [name for name, _, _ in packed],
            ["drafter", "target-0", "target-1", "target-2"],
        )
        self.assertEqual([gpu for _, gpu, _ in packed], [3, 0, 1, 2])
        self.assertTrue(all("DECOUPLED" in cmd for _, _, cmd in packed[1:]))
        self.assertTrue(
            all("--speculative-draft-model-path" in cmd for _, _, cmd in packed[1:])
        )
        drafter_cmd = packed[0][2]
        self.assertIn("engine", drafter_cmd)
        self.assertNotIn("--http-url", drafter_cmd)
        self.assertNotIn("dllm-draft-server", [name for name, _, _ in packed])
        self.assertIn("ipc:///dev/shm/sglang-decoupled-32000.sock", drafter_cmd)
        self.assertEqual(
            drafter_cmd[drafter_cmd.index("--max-sessions") + 1], "144"
        )
        self.assertEqual(
            drafter_cmd[drafter_cmd.index("--max-sessions-per-peer") + 1], "48"
        )
        self.assertEqual(
            drafter_cmd[
                drafter_cmd.index("--engine-max-running-requests") + 1
            ],
            "208",
        )
        self.assertEqual(
            drafter_cmd[
                drafter_cmd.index("--engine-packed-cuda-graph-max-requests") + 1
            ],
            "8",
        )
        self.assertEqual(
            drafter_cmd[
                drafter_cmd.index("--engine-packed-cuda-graph-max-tokens") + 1
            ],
            "1024",
        )
        self.assertNotIn("--quantization", drafter_cmd)

        args.remote_draft_quantization = "fp8"
        quantized = module.commands(args)
        self.assertEqual(
            quantized[0][2][quantized[0][2].index("--quantization") + 1],
            "fp8",
        )
        self.assertTrue(
            all("--quantization" not in cmd for _, _, cmd in quantized[1:])
        )
        self.assertTrue(
            all("--speculative-draft-model-quantization" not in cmd
                for _, _, cmd in quantized[1:])
        )
        args.remote_draft_quantization = None

        args.remote_draft_model = "draft-int8"
        args.remote_draft_quantization = "w8a8_int8"
        quantized = module.commands(args)
        self.assertEqual(
            quantized[0][2][quantized[0][2].index("--model-path") + 1],
            "draft-int8",
        )
        self.assertEqual(
            quantized[0][2][quantized[0][2].index("--quantization") + 1],
            "w8a8_int8",
        )
        self.assertTrue(
            all(
                cmd[cmd.index("--speculative-draft-model-path") + 1] == "draft"
                for _, _, cmd in quantized[1:]
            )
        )
        args.remote_draft_model = None
        args.remote_draft_quantization = None

        args.num_targets = 1
        args.gpus = [0, 3]
        one_plus_one = module.commands(args)
        self.assertEqual(
            [(name, gpu) for name, gpu, _ in one_plus_one],
            [("drafter", 3), ("target-0", 0)],
        )
        one_drafter_cmd = one_plus_one[0][2]
        self.assertEqual(
            one_drafter_cmd[one_drafter_cmd.index("--max-sessions") + 1], "48"
        )
        args.wait_for_covered_result = True
        self.assertIn(
            "--decoupled-wait-for-covered-result", module.commands(args)[1][2]
        )
        args.wait_for_covered_result = False
        args.mode = "local"
        args.gpus = [0]
        self.assertEqual(
            [(name, gpu) for name, gpu, _ in module.commands(args)],
            [("target-0", 0)],
        )
        args.remote_draft_quantization = "fp8"
        with self.assertRaisesRegex(ValueError, "requires hybrid mode"):
            module.commands(args)
        args.remote_draft_quantization = None
        args.remote_draft_model = "draft-int8"
        with self.assertRaisesRegex(ValueError, "requires hybrid mode"):
            module.commands(args)
        args.remote_draft_model = None

        args.mode = "hybrid"
        args.gpus = [0, 3]
        args.dllm_algorithm = "LowConfidence"
        with self.assertRaisesRegex(ValueError, "requires.*OneStep"):
            module.commands(args)


class TestAggregateBenchmark(CustomTestCase):
    def test_dataset_accepts_math500_problem_schema(self):
        class Tokenizer:
            def encode(self, text):
                return [len(text)]

            def apply_chat_template(self, messages, **kwargs):
                self.template_call = (messages, kwargs)
                return {"input_ids": [99, len(messages[0]["content"])]}

        tokenizer = Tokenizer()
        prompts = encode_dataset_prompts(
            rows=[
                {"input_ids": [1, 2]},
                {"text": "text"},
                {"problem": "math500"},
            ],
            tokenizer=tokenizer,
            limit=3,
        )
        self.assertEqual(prompts, [[1, 2], [4], [99, 78]])
        messages, kwargs = tokenizer.template_call
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Please reason step by step", messages[0]["content"])
        self.assertEqual(
            kwargs,
            {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        )

    def test_common_wall_time_and_failures(self):
        summary = summarize(
            results=[
                Result(0, 0, 1, 100),
                Result(1, 1, 2, 100),
                Result(2, 2, 3, 0, "timeout"),
            ],
            elapsed=4,
        )
        self.assertEqual(summary["output_tokens_per_second"], 50)
        self.assertEqual(summary["requests_per_second"], 0.5)
        self.assertFalse(summary["valid"])

    def test_same_text_with_different_tokens_cannot_pass(self):
        reference = {
            "prompt_digest": "fixed",
            "configuration": {"model_path": "target", "output_len": 2},
            "summary": {"valid": True},
            "requests": [
                {"index": 0, "output_ids": [1, 2], "output_tokens": 2, "error": ""}
            ],
        }
        candidate = copy.deepcopy(reference)
        self.assertTrue(
            compare_outputs(payload=candidate, reference=reference)["valid"]
        )
        candidate["requests"][0]["output_ids"] = (1, 2)
        self.assertTrue(
            compare_outputs(payload=candidate, reference=reference)["valid"]
        )
        candidate["requests"][0]["output_ids"] = [1, 3]
        candidate["requests"][0]["output_top_logprobs"] = [
            [[-0.1, 1, None], [-2.0, 4, None]],
            [[-0.6, 3, None], [-0.7, 2, None]],
        ]
        reference["requests"][0]["output_top_logprobs"] = [
            [[-0.1, 1, None], [-2.0, 4, None]],
            [[-0.65, 2, None], [-0.65, 3, None]],
        ]
        comparison = compare_outputs(payload=candidate, reference=reference)
        self.assertEqual(comparison["mismatched_indices"], [0])
        self.assertEqual(
            comparison["divergence_details"],
            [
                {
                    "index": 0,
                    "first_token": 1,
                    "reference_tokens": [2],
                    "current_tokens": [3],
                    "reference_top_logprobs": [
                        [-0.65, 2, None],
                        [-0.65, 3, None],
                    ],
                    "current_top_logprobs": [
                        [-0.6, 3, None],
                        [-0.7, 2, None],
                    ],
                }
            ],
        )
        candidate["prompt_digest"] = "different"
        with self.assertRaises(ValueError):
            compare_outputs(payload=candidate, reference=reference)

    def test_http_workload_is_global_and_warmups_are_excluded(self):
        async def exercise():
            received = []

            async def generate(request):
                data = await request.json()
                received.append(data)
                count = data["sampling_params"]["max_new_tokens"]
                return web.json_response(
                    {
                        "output_ids": [5] * count,
                        "meta_info": {
                            "completion_tokens": count,
                            "output_top_logprobs": [
                                [[0.0, 5, None], [-1.0, 4, None]]
                            ]
                            * count,
                        },
                    }
                )

            app = web.Application()
            app.router.add_post("/generate", generate)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            try:
                args = SimpleNamespace(
                    timeout=5,
                    worker_urls=[f"http://127.0.0.1:{port}"] * 3,
                    output_len=2,
                    request_rate=float("inf"),
                    seed=42,
                    divergence_top_logprobs=2,
                )
                result = await run(args, [[1], [2], [3], [4]])
                self.assertEqual(result["summary"]["completed_requests"], 4)
                self.assertEqual(result["summary"]["output_tokens"], 8)
                self.assertEqual(len(received), 7)
                self.assertEqual(received[0]["input_ids"], [1])
                self.assertEqual(received[0]["top_logprobs_num"], 2)
                self.assertEqual(
                    [r["worker"] for r in result["requests"]], [0, 1, 2, 0]
                )
            finally:
                await runner.cleanup()

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
