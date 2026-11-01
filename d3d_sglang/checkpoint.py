"""Adapt published Qwen3 draft metadata to the vendored diffusion engine."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile


@contextmanager
def draft_checkpoint(model_path):
    path = Path(model_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_path))
    config = json.loads((path / "config.json").read_text())
    if not config.get("d3d_draft"):
        yield str(path)
        return
    if config.get("model_type") != "qwen3":
        raise ValueError("D3D draft must use the Qwen3 architecture")
    config["architectures"] = ["D3DForCausalLM"]
    config.pop("auto_map", None)
    with tempfile.TemporaryDirectory(prefix="d3d-engine-") as directory:
        staged = Path(directory)
        for source in path.iterdir():
            if source.name != "config.json":
                (staged / source.name).symlink_to(source.resolve())
        (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        yield str(staged)
