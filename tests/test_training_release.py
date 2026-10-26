import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TrainingLauncherTests(unittest.TestCase):
    def test_training_requirements_reuse_native_inference_dependencies(self):
        requirements = (ROOT / 'requirements-train.txt').read_text().splitlines()
        self.assertEqual(requirements, ['-r requirements.txt'])

    def run_launcher(self, script, arguments, environment=None):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / 'arguments.json'
            executable = Path(directory) / 'python-stub'
            executable.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'from pathlib import Path\n'
                'Path(os.environ["D3D_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
            )
            executable.chmod(0o755)
            launch_environment = dict(os.environ)
            launch_environment.update(
                PY=str(executable), D3D_CAPTURE=str(capture),
                NUM_GPUS='2', DRAFT_MODEL='checkpoints/initial',
            )
            launch_environment.update(environment or {})
            result = subprocess.run(
                ['bash', str(ROOT / 'scripts' / script), *arguments],
                cwd=directory, env=launch_environment, capture_output=True, text=True,
            )
            captured = json.loads(capture.read_text()) if capture.exists() else None
            return result, captured

    def test_training_launcher_selects_training_code_and_forwards_overrides(self):
        result, arguments = self.run_launcher(
            'train.sh', ['--block-size', '32', '--disable-wandb'],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments[:2], ['-m', 'torch.distributed.run'])
        self.assertEqual(arguments[arguments.index('--nproc_per_node') + 1], '2')
        self.assertEqual(arguments[arguments.index('--module') + 1], 'train.train')
        self.assertEqual(
            Path(arguments[arguments.index('--draft-code-model') + 1]),
            ROOT / 'models' / 'd3d_training',
        )
        self.assertEqual(arguments[-3:], ['--block-size', '32', '--disable-wandb'])

    def test_pruning_launcher_routes_all_stages(self):
        for stage, module in (
            ('select', 'pruning.select_layer'), ('prune', 'pruning.prune'),
            ('choose', 'pruning.select_checkpoint'),
        ):
            with self.subTest(stage=stage):
                result, arguments = self.run_launcher('prune.sh', [stage, '--help'])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(arguments, ['-m', module, '--help'])
        result, arguments = self.run_launcher(
            'prune.sh', ['train', '--nproc-per-node', '2', '--help'],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments[arguments.index('--module') + 1], 'pruning.train_recovery')
        self.assertEqual(arguments[-1], '--help')

    def test_pruning_launcher_rejects_invalid_worker_counts(self):
        for worker_count in ('0', '-1', 'two'):
            with self.subTest(worker_count=worker_count):
                result, arguments = self.run_launcher(
                    'prune.sh', ['train', '--nproc-per-node', worker_count, '--help'],
                )
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(arguments)

    def test_mask_token_validation(self):
        from models.token_ids import checkpoint_mask_id

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps({
                'vocab_size': 64, 'mask_token_id': 63,
            }))
            self.assertEqual(checkpoint_mask_id(directory), 63)
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                checkpoint_mask_id(directory, 62)
            (Path(directory) / 'config.json').write_text(json.dumps({'vocab_size': 64}))
            with self.assertRaisesRegex(ValueError, 'outside'):
                checkpoint_mask_id(directory)


try:
    TRAINING_ENVIRONMENT = importlib.metadata.version('transformers').startswith('4.57.')
except importlib.metadata.PackageNotFoundError:
    TRAINING_ENVIRONMENT = False


@unittest.skipUnless(TRAINING_ENVIRONMENT, 'requires the shared Transformers 4.57 environment')
class TrainingRuntimeTests(unittest.TestCase):
    def test_pruning_remaps_weights_and_loads_training_model(self):
        import torch
        from models.d3d_training.configuration_d3d import D3DConfig
        from models.d3d_training.modeling_d3d import D3DForCausalLM
        from pruning.modeling import load_drafter
        from pruning.prune import prune

        config = D3DConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, pad_token_id=0, bos_token_id=1,
            eos_token_id=2, use_sliding_window=False,
        )
        config.mask_token_id = 63
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / 'parent'
            child = Path(directory) / 'child'
            source_model = D3DForCausalLM(config)
            source_model.save_pretrained(parent)
            selection = Path(directory) / 'selection.json'
            selection.write_text(json.dumps({
                'parent_checkpoint': str(parent), 'teacher_checkpoint': str(parent),
                'parent_depth': 2, 'child_depth': 1, 'original_depth': 2,
                'removed_compact_layer': 0, 'removed_original_layer': 0,
                'child_retained_original_layers': [1], 'block_size': 16,
                'mask_token_id': 63,
            }))
            prune(selection, child)
            model = load_drafter(str(child), torch.float32)
            self.assertEqual(len(model.model.layers), 1)
            self.assertEqual(model.config.mask_token_id, 63)
            torch.testing.assert_close(
                model.model.layers[0].self_attn.q_proj.weight,
                source_model.model.layers[1].self_attn.q_proj.weight,
            )
            self.assertEqual(
                json.loads((child / 'layer_prune.json').read_text())['retained_original_layers'],
                [1],
            )
            with self.assertRaises(FileExistsError):
                prune(selection, child)

    def test_default_code_model_is_training_port(self):
        from train.arguments import parse_args

        with patch.object(sys, 'argv', [
            'train', '--draft-model', 'draft', '--data-path', 'data.jsonl',
            '--output-dir', 'output',
        ]):
            arguments = parse_args()
        self.assertEqual(Path(arguments.draft_code_model), ROOT / 'models/d3d_training')

    def test_staging_preserves_architecture_and_original_checkpoint(self):
        from train.runtime import stage_checkpoint_with_code_model

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            original = {'model_type': 'qwen3', 'hidden_size': 32, 'num_hidden_layers': 2}
            (checkpoint / 'config.json').write_text(json.dumps(original))
            (checkpoint / 'model.safetensors').write_bytes(b'weight-placeholder')
            with stage_checkpoint_with_code_model(
                directory, str(ROOT / 'models/d3d_training'),
            ) as staged:
                staged_config = json.loads((Path(staged) / 'config.json').read_text())
                self.assertEqual(staged_config['model_type'], 'd3d')
                self.assertEqual(staged_config['hidden_size'], 32)
                self.assertEqual(staged_config['num_hidden_layers'], 2)
                self.assertIn('AutoModelForCausalLM', staged_config['auto_map'])
                self.assertTrue((Path(staged) / 'modeling_d3d.py').is_file())
                self.assertTrue((Path(staged) / 'model.safetensors').is_symlink())
            self.assertEqual(json.loads((checkpoint / 'config.json').read_text()), original)
            self.assertFalse((checkpoint / 'modeling_d3d.py').exists())

    def test_dataset_normalization_and_generation_status(self):
        import torch
        from train.dataset import _normalize_messages, encode_offline_example

        self.assertEqual(_normalize_messages([
            {'from': 'human', 'value': 'question'},
            {'from': 'gpt', 'value': 'answer', 'reasoning_content': 'reason'},
        ]), [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'answer', 'reasoning_content': 'reason'},
        ])
        self.assertIsNone(encode_offline_example(
            None, {'status': 'length_truncated'}, fallback_prompt_id=0,
            max_prompt_len=32, max_seq_len=64, enable_thinking=False,
            device=torch.device('cpu'),
        ))

    def test_tiny_packed_training_forward_and_backward(self):
        import torch
        from models.d3d_training.configuration_d3d import D3DConfig
        from models.d3d_training.modeling_d3d import D3DForCausalLM
        from train.blocks import build_packed_block_batch
        from train.losses import weighted_objective_loss_sum
        from train.model_utils import packed_draft_forward, prefill_clean_kv_cache
        from train.types import OfflineExample

        config = D3DConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, pad_token_id=0, bos_token_id=1,
            eos_token_id=2, use_sliding_window=False,
        )
        config._attn_implementation = 'sdpa'
        model = D3DForCausalLM(config).train()
        example = OfflineExample(
            prompt_id='tiny', prompt_ids=torch.tensor([[3, 4]]),
            response_ids=torch.tensor([[5, 6, 7, 8]]),
        )
        batch = build_packed_block_batch(
            example, block_size=4, mask_token_id=63, max_anchors=3,
            device=torch.device('cpu'),
        )
        self.assertIsNotNone(batch)
        self.assertEqual(batch.active_lengths.tolist(), [[3, 2, 1]])
        cache = prefill_clean_kv_cache(model, batch.clean_input_ids, batch.clean_lens)
        output = packed_draft_forward(
            model, batch, clean_cache=cache, gradient_checkpointing=False,
        )
        loss = weighted_objective_loss_sum(
            output.logits[batch.active_token_mask], batch.target_ids[batch.active_token_mask],
            objective='ce', loss_weights=torch.ones(int(batch.active_token_mask.sum())),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.model.layers[0].self_attn.q_proj.weight.grad)
        self.assertTrue(torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
