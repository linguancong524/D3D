import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PublicReleaseTests(unittest.TestCase):
    def test_benchmark_scripts_forward_token_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / 'arguments.jsonl'
            executable = root / 'python-stub'
            executable.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'if sys.argv[1:2] != ["-c"]:\n'
                '    with open(os.environ["D3D_CAPTURE"], "a") as stream:\n'
                '        stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            )
            executable.chmod(0o755)
            for script in ('benchmark.sh', 'benchmark_concurrent.sh'):
                for token_limit in (None, '128'):
                    with self.subTest(script=script, token_limit=token_limit):
                        environment = dict(os.environ)
                        environment.update(
                            PYTHON_BIN=str(executable), D3D_CAPTURE=str(capture),
                            DATASETS='math500', CONCURRENCY='4',
                        )
                        command = ['bash', str(ROOT / 'scripts' / script)]
                        if token_limit is not None:
                            command += ['--max-new-tokens', token_limit]
                        result = subprocess.run(
                            command, cwd=root, env=environment, capture_output=True, text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        arguments = json.loads(capture.read_text().splitlines()[-1])
                        # The script passes its default first; appended CLI flags come last.
                        index = max(i for i, value in enumerate(arguments)
                                    if value == '--max-new-tokens')
                        self.assertEqual(arguments[index + 1], token_limit or '2048')

    def test_standard_library_concurrent_is_not_shadowed(self):
        result = subprocess.run(
            [sys.executable, '-c', 'import concurrent.futures'],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_concurrent_benchmark_picks_anchor_by_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / 'arguments.jsonl'
            executable = root / 'python-stub'
            executable.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'if sys.argv[1:2] != ["-c"]:\n'
                '    with open(os.environ["D3D_CAPTURE"], "a") as stream:\n'
                '        stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            )
            executable.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                PYTHON_BIN=str(executable), D3D_CAPTURE=str(capture),
                DATASETS='math500', CONCURRENCY='1 4',
            )
            result = subprocess.run(
                ['bash', str(ROOT / 'scripts' / 'benchmark_concurrent.sh')],
                cwd=root, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            invocations = [json.loads(line) for line in capture.read_text().splitlines()]
            concurrencies = [args[args.index('--concurrency') + 1] for args in invocations]
            anchors = [args[args.index('--anchor-num') + 1] for args in invocations]
            self.assertEqual(concurrencies, ['1', '4'])
            self.assertEqual(anchors, ['3', '1'])

    def test_local_checkpoint_needs_no_network(self):
        utils = importlib.import_module('common.runtime')
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps({'model_type': 'd3d'}))
            self.assertEqual(utils.resolve_model_path(directory), directory)

    def test_hub_checkpoint_is_resolved_to_snapshot(self):
        utils = importlib.import_module('common.runtime')
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text('{}')
            module = types.ModuleType('huggingface_hub')
            module.snapshot_download = lambda **kwargs: directory
            with patch.dict(sys.modules, {'huggingface_hub': module}):
                self.assertEqual(utils.resolve_model_path('example/draft'), directory)

    def test_missing_local_checkpoint_fails_clearly(self):
        utils = importlib.import_module('common.runtime')
        with self.assertRaises(FileNotFoundError):
            utils.resolve_model_path('/missing/d3d-checkpoint')

    def test_missing_cuda_fails_before_model_download(self):
        utils = importlib.import_module('common.runtime')
        module = types.ModuleType('torch')
        module.cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {'torch': module}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                utils.check_hardware()

    def test_same_device_is_rejected(self):
        utils = importlib.import_module('common.runtime')
        module = types.ModuleType('torch')
        module.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        with patch.dict(sys.modules, {'torch': module}):
            with self.assertRaisesRegex(ValueError, 'distinct'):
                utils.check_hardware('cuda:0', 'cuda:0')

    def test_dataset_jsonl_download_preserves_source_rows(self):
        utils = importlib.import_module('common.data')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'test6.jsonl'
            source.write_text('{"question_content": "solve", "starter_code": "def solve():"}\n')
            module = types.ModuleType('huggingface_hub')
            module.hf_hub_download = lambda **kwargs: str(source)
            with patch.dict(sys.modules, {'huggingface_hub': module}):
                rows, metadata = utils.load_dataset_jsonl('example/data', 'test6.jsonl')
            self.assertEqual(rows[0]['starter_code'], 'def solve():')
            self.assertEqual(metadata['filename'], 'test6.jsonl')
            self.assertEqual(len(metadata['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
