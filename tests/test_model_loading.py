import tempfile
import json
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class ModelLoadingTests(unittest.TestCase):
    def test_d3d_mlp_cpu_forward_with_optional_liger_installed(self):
        from models.d3d.configuration_d3d import D3DConfig
        from models.d3d import modeling_d3d

        config = D3DConfig(hidden_size=32, intermediate_size=64)
        model = modeling_d3d.D3DMLP(config)
        inputs = torch.randn(1, 2, 32)
        expected = model.down_proj(model.act_fn(model.gate_proj(inputs)) * model.up_proj(inputs))
        with patch.object(modeling_d3d, 'liger_kernel_is_available', True):
            actual = model(inputs)
        torch.testing.assert_close(actual, expected)

    def test_d3d_mlp_liger_on_noncurrent_gpu(self):
        from models.d3d.configuration_d3d import D3DConfig
        from models.d3d import modeling_d3d

        if torch.cuda.device_count() < 2 or not modeling_d3d.liger_kernel_is_available:
            self.skipTest('Requires two CUDA devices and Liger kernels')
        config = D3DConfig(hidden_size=32, intermediate_size=64)
        with torch.cuda.device(0), torch.inference_mode():
            model = modeling_d3d.D3DMLP(config).to(device='cuda:1', dtype=torch.bfloat16)
            inputs = torch.randn(1, 2, 32, device='cuda:1', dtype=torch.bfloat16)
            expected = model.down_proj(model.act_fn(model.gate_proj(inputs)) * model.up_proj(inputs))
            actual = model(inputs)
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
            self.assertEqual(torch.cuda.current_device(), 0)

    def test_model_loader_does_not_execute_remote_repository_code(self):
        from single.eval_sync import _from_pretrained

        class FakeModel:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return kwargs

        kwargs = _from_pretrained(FakeModel, 'local-model', torch.bfloat16,
                                  attn_implementation=None)
        self.assertFalse(kwargs['trust_remote_code'])

    def test_serving_loader_rejects_empty_weights(self):
        from serving.vendor.ssd.utils.loader import load_safetensors_model

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'safetensors'):
                load_safetensors_model(torch.nn.Linear(2, 2), directory, {})

    def test_serving_loader_rejects_missing_parameter(self):
        from safetensors.torch import save_file
        from serving.vendor.ssd.utils.loader import load_safetensors_model

        with tempfile.TemporaryDirectory() as directory:
            save_file({'weight': torch.ones(2, 2)}, str(Path(directory) / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'bias'):
                load_safetensors_model(torch.nn.Linear(2, 2), directory, {})

    def test_serving_loader_accepts_complete_checkpoint(self):
        from safetensors.torch import save_file
        from serving.vendor.ssd.utils.loader import load_safetensors_model

        with tempfile.TemporaryDirectory() as directory:
            save_file({'weight': torch.ones(2, 2), 'bias': torch.ones(2)},
                      str(Path(directory) / 'model.safetensors'))
            model = torch.nn.Linear(2, 2)
            load_safetensors_model(model, directory, {})
            self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))
            self.assertTrue(torch.equal(model.bias, torch.ones(2)))

    def test_serving_loader_requires_every_packed_projection(self):
        from safetensors.torch import save_file
        from serving.vendor.ssd.utils.loader import load_safetensors_model

        class PackedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = torch.nn.Linear(2, 6, bias=False)
                self.qkv_proj.weight.weight_loader = lambda param, data, shard: None

        mapping = {'q_proj': ('qkv_proj', 'q'), 'k_proj': ('qkv_proj', 'k'),
                   'v_proj': ('qkv_proj', 'v')}
        with tempfile.TemporaryDirectory() as directory:
            save_file({'q_proj.weight': torch.ones(2, 2),
                       'k_proj.weight': torch.ones(2, 2)},
                      str(Path(directory) / 'model.safetensors'))
            with self.assertRaisesRegex(ValueError, 'qkv_proj.weight'):
                load_safetensors_model(PackedModel(), directory, mapping)

    def test_serving_loader_rejects_missing_index_shard(self):
        import json
        from safetensors.torch import save_file
        from serving.vendor.ssd.utils.loader import load_safetensors_model

        with tempfile.TemporaryDirectory() as directory:
            save_file({'weight': torch.ones(2, 2), 'bias': torch.ones(2)},
                      str(Path(directory) / 'part-1.safetensors'))
            index = {'weight_map': {'weight': 'part-1.safetensors',
                                    'bias': 'part-2.safetensors'}}
            (Path(directory) / 'model.safetensors.index.json').write_text(json.dumps(index))
            with self.assertRaisesRegex(FileNotFoundError, 'part-2.safetensors'):
                load_safetensors_model(torch.nn.Linear(2, 2), directory, {})

    def test_local_d3d_checkpoint_roundtrip_uses_bfloat16(self):
        from models.d3d.configuration_d3d import D3DConfig
        from models.d3d.modeling_d3d import D3DForCausalLM
        from single.eval_sync import _load_model

        config = D3DConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
            bos_token_id=2, use_sliding_window=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = D3DForCausalLM(config)
            source.save_pretrained(directory)
            config_path = Path(directory) / 'config.json'
            serialized = json.loads(config_path.read_text())
            self.assertEqual(serialized['model_type'], 'd3d')
            self.assertEqual(serialized['architectures'], ['D3DForCausalLM'])
            for model_type in ('d3d', 'sdar'):
                with self.subTest(model_type=model_type):
                    serialized['model_type'] = model_type
                    config_path.write_text(json.dumps(serialized))
                    model = _load_model(directory, torch.bfloat16, attn_implementation='sdpa')
                    self.assertEqual(next(model.parameters()).dtype, torch.bfloat16)
                    self.assertEqual(set(source.state_dict()), set(model.state_dict()))
                    for name, tensor in source.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[name], tensor.bfloat16(),
                                                   rtol=0, atol=0)
