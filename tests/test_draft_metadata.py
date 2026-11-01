import json
from pathlib import Path
import tempfile
import unittest

from d3d_sglang.checkpoint import draft_checkpoint
from single.eval_sync import _checkpoint_model_type


class DraftMetadataTests(unittest.TestCase):
    def test_qwen_target_is_not_a_diffusion_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({
                'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM'],
            }))
            self.assertEqual(_checkpoint_model_type(directory), 'qwen3')
            with draft_checkpoint(directory) as staged:
                self.assertEqual(staged, directory)

    def test_engine_adaptation_preserves_published_config_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = {'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM'],
                      'd3d_draft': True, 'hidden_size': 32,
                      'auto_map': {'AutoConfig': 'configuration_d3d.D3DConfig'}}
            (path / 'config.json').write_text(json.dumps(config))
            (path / 'model.safetensors').write_bytes(b'weights')
            self.assertEqual(_checkpoint_model_type(directory), 'd3d')
            with draft_checkpoint(directory) as staged:
                staged = Path(staged)
                adapted = json.loads((staged / 'config.json').read_text())
                self.assertEqual(adapted['architectures'], ['D3DForCausalLM'])
                self.assertNotIn('auto_map', adapted)
                self.assertEqual(adapted['hidden_size'], 32)
                self.assertEqual((staged / 'model.safetensors').resolve(), path / 'model.safetensors')
                self.assertEqual(json.loads((path / 'config.json').read_text()), config)
            self.assertFalse(staged.exists())
