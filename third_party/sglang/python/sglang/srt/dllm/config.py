from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


DLLM_PARAMS = {
    "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
    "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
    "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
}


def resolve_dllm_block_size(
    *, architecture: str, algorithm_config: dict[str, Any]
) -> int:
    if architecture not in DLLM_PARAMS:
        raise RuntimeError(f"Unknown diffusion LLM: {architecture}")
    block_size = algorithm_config.get(
        "block_size", DLLM_PARAMS[architecture]["block_size"]
    )
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("dLLM block_size must be an integer")
    if block_size < 1:
        raise ValueError("dLLM block_size must be positive")
    return block_size


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        first_done_first_out_mode: bool = False,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.first_done_first_out_mode = first_done_first_out_mode
        # OneStep is the SDAR/DECOUPLED path: the target-produced anchor is the
        # first position of the denoising block. Existing iterative dLLM
        # algorithms keep their upstream all-mask, fixed-width block contract.
        self.anchor_conditioned_block = algorithm == "OneStep"

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision or "",
        )
        architectures = model_config.hf_config.architectures
        if not architectures:
            raise RuntimeError("Diffusion LLM config has no architecture")
        arch = architectures[0]
        if arch not in DLLM_PARAMS:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")
        mask_id = DLLM_PARAMS[arch]["mask_id"]

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            try:
                with open(server_args.dllm_algorithm_config, "r") as f:
                    algorithm_config = yaml.safe_load(f) or {}
            except OSError as exc:
                raise ValueError(
                    "Failed to read dLLM algorithm config: "
                    f"{server_args.dllm_algorithm_config}"
                ) from exc

        block_size = resolve_dllm_block_size(
            architecture=arch, algorithm_config=algorithm_config
        )

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            first_done_first_out_mode=server_args.dllm_fdfo,
        )
