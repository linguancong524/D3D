"""Offline SDAR INT8 checkpoint conversion invariants."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSdarInt8Checkpoint(CustomTestCase):
    def test_per_channel_scales_and_unquantized_vocab_head(self):
        """A missing scale or quantized vocab head would break SDAR loading."""
        root = Path(__file__).resolve().parents[4]
        spec = importlib.util.spec_from_file_location(
            "quantize_sdar_w8a8_int8",
            root / "scripts/decoupled/quantize_sdar_w8a8_int8.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "int8"
            source.mkdir()
            weights = {
                "model.layers.0.self_attn.q_proj.weight": torch.tensor(
                    [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
                ),
                "lm_head.weight": torch.ones((4, 3)),
            }
            shard_name = "model-00001-of-00001.safetensors"
            save_file(weights, source / shard_name)
            (source / "config.json").write_text(json.dumps({"model_type": "sdar"}))
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 72},
                        "weight_map": {name: shard_name for name in weights},
                    }
                )
            )

            summary = module.convert_checkpoint(source=source, output=output)
            converted = load_file(output / shard_name)
            name = "model.layers.0.self_attn.q_proj"
            self.assertEqual(summary["num_quantized_linear_weights"], 1)
            self.assertEqual(converted[name + ".weight"].dtype, torch.int8)
            self.assertEqual(converted[name + ".weight_scale"].shape, (2, 1))
            self.assertEqual(converted["lm_head.weight"].dtype, torch.float32)
            self.assertTrue(
                torch.allclose(
                    converted[name + ".weight"].float()
                    * converted[name + ".weight_scale"],
                    weights[name + ".weight"],
                    atol=0.01,
                )
            )
            config = json.loads((output / "config.json").read_text())
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(config["quantization_config"]["quant_method"], "w8a8_int8")
            self.assertEqual(index["weight_map"][name + ".weight_scale"], shard_name)
            with self.assertRaises(FileExistsError):
                module.convert_checkpoint(source=source, output=output)


if __name__ == "__main__":
    unittest.main()
