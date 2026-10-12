import importlib.metadata
import hashlib
import json
import platform
import subprocess
from math import ceil
from pathlib import Path


def resolve_model_path(model, revision=None):
    path = Path(model).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
    else:
        if path.is_absolute() or str(model).startswith(('.', '~')):
            raise FileNotFoundError(f'Model directory does not exist: {model}')
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(
            repo_id=str(model), revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.bin', '*.model', '*.txt', '*.tiktoken'],
        )
    if not (Path(resolved) / 'config.json').is_file():
        raise FileNotFoundError(f'Missing config.json in checkpoint: {resolved}')
    return resolved


def check_hardware(target_device='cuda:0', draft_device='cuda:1'):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use a CUDA-enabled PyTorch build and two NVIDIA GPUs.')
    if target_device == draft_device:
        raise ValueError('Target and draft must use two distinct CUDA devices.')
    for name in (target_device, draft_device):
        device = torch.device(name)
        if device.type != 'cuda' or device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f'Unavailable CUDA device: {name}; select two visible GPUs.')
        if torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError(f'{name} lacks the required native bfloat16 support.')


def check_paged_capacity(target, draft, datasets, concurrency, max_new_tokens, pages,
                         block_size=16, anchor_num=3):
    draft_branch_tokens = block_size * (2 + (block_size - 1) * anchor_num)
    for name, path, extra in (('target', target, block_size),
                              ('draft', draft, draft_branch_tokens)):
        config = json.loads((Path(path) / 'config.json').read_text())
        context_limit = min(config['max_position_embeddings'], 16384)
        for dataset, prompts in datasets.items():
            if not prompts:
                continue
            required_tokens = max(map(len, prompts)) + max_new_tokens + extra
            if required_tokens > context_limit:
                raise ValueError(f'{name} context limit {context_limit} is too small for '
                                 f'{dataset}: need {required_tokens} tokens including branches')
            for count in concurrency:
                required_pages = min(count, len(prompts)) * ceil(required_tokens / 256)
                if required_pages > pages:
                    raise ValueError(f'{name} needs {required_pages} pages for {dataset} '
                                     f'at concurrency {count}; configured pages={pages}')


def load_dataset_jsonl(repo_id, filename, revision='main'):
    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset', revision=revision))
    content = source.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    return rows, {'repo_id': repo_id, 'filename': filename, 'revision': revision,
                  'sha256': hashlib.sha256(content).hexdigest()}


def runtime_metadata(models=None):
    import torch

    versions = {}
    for package in ('torch', 'transformers', 'datasets', 'flash-attn', 'triton', 'huggingface-hub'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[1]
    commit = None
    if (root / '.git').exists():
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if result.returncode == 0:
            commit = result.stdout.strip()
    checkpoints = {}
    for name, model in (models or {}).items():
        path = Path(model)
        config = json.loads((path / 'config.json').read_text())
        checkpoints[name] = {
            'model_type': config.get('model_type'),
            'revision': path.name if path.parent.name == 'snapshots' else config.get('_commit_hash'),
        }
    return {
        'python': platform.python_version(), 'packages': versions, 'cuda_build': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        'source_commit': commit, 'checkpoints': checkpoints,
        'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(root.rglob('*.py'))},
    }
