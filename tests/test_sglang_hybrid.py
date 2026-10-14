import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("d3d_hybrid", ROOT / "d3d_sglang/hybrid.py")
HYBRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HYBRID)


class HybridLauncherTests(unittest.TestCase):
    def test_concurrency_reserves_padding_slot_and_keeps_graph_enabled(self):
        for concurrency in (1, 4, 8, 16):
            specs = {"commands": [
                ["drafter", 1, ["python", "-m", "sglang.launch_drafter"]],
                ["target-0", 0, ["python", "-m", "sglang.launch_server",
                                  "--max-running-requests", str(concurrency),
                                  "--enable-deterministic-inference"]],
            ]}
            with patch.object(HYBRID.subprocess, "check_output", return_value=json.dumps(specs)) as output:
                result = HYBRID.commands({"MAX_RUNNING_REQUESTS": str(concurrency)}, [])
            upstream = output.call_args.args[0]
            self.assertIn("--draft-enable-cuda-graph", upstream)
            self.assertEqual(upstream[upstream.index("--num-anchors") + 1], "1")
            self.assertEqual(upstream[upstream.index("--local-cuda-graph-max-requests") + 1], str(concurrency))
            target = result["commands"][1][2]
            self.assertEqual(target[target.index("--max-running-requests") + 1], str(concurrency + 1))
            self.assertEqual(target[target.index("--cuda-graph-max-bs-decode") + 1], str(concurrency))
            self.assertNotIn("--enable-deterministic-inference", target)
            self.assertNotIn("--cuda-graph-backend-decode=disabled", target)

    def test_deterministic_target_flag_is_opt_in(self):
        specs = {"commands": [["target-0", 0, [
            "python", "--max-running-requests", "1", "--enable-deterministic-inference",
        ]]]}
        with patch.object(HYBRID.subprocess, "check_output", return_value=json.dumps(specs)):
            result = HYBRID.commands({"DETERMINISTIC": "1"}, [])
        self.assertIn("--enable-deterministic-inference", result["commands"][0][2])

    def test_invalid_concurrency_is_rejected_before_launch(self):
        with self.assertRaises(ValueError):
            HYBRID.commands({"MAX_RUNNING_REQUESTS": "0"}, [])
