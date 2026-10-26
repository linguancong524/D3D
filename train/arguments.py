from __future__ import annotations

import argparse
from pathlib import Path


from .runtime import add_d3d_fsdp_args

def parse_args(*, default_fix_mistral_regex: bool = False) -> argparse.Namespace:
    """Parse CLI arguments for offline D3D OPD training."""

    parser = argparse.ArgumentParser(description="Offline D3D OPD on teacher-generated Qwen response tokens.")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-code-model", default=str(Path(__file__).resolve().parents[1] / "models/d3d_training"))
    parser.add_argument("--trajectory-teacher-model", default="")
    parser.add_argument("--trajectory-teacher-code-model", default="")
    parser.add_argument("--trajectory-weight", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-gradient-probe-steps",
        type=int,
        default=0,
        help=(
            "For the first N micro-steps, measure unit trajectory versus CE "
            "gradient norm at the captured final hidden. Zero disables it."
        ),
    )
    parser.add_argument(
        "--trajectory-shuffle-teacher-blocks",
        action="store_true",
        help=(
            "Negative control: shuffle teacher relation targets among selected "
            "blocks with identical active length."
        ),
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-model", default="")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--objective", default="ce")
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--hnce-weight", type=float, default=0.0)
    parser.add_argument("--hnce-top-k", type=int, default=16)
    parser.add_argument("--hnce-margin", type=float, default=0.0)
    parser.add_argument("--margin-weight", type=float, default=0.0)
    parser.add_argument("--margin-top-k", type=int, default=4)
    parser.add_argument("--margin-value", type=float, default=0.5)
    parser.add_argument("--margin-smooth", action="store_true")
    parser.add_argument("--ar-loss-weight", type=float, default=0.0)
    parser.add_argument("--ar-logit-chunk-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--loss-decay-gamma", type=float, default=15.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parallel-anchors-per-example", type=int, default=32)
    parser.add_argument("--max-prompt-len", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-micro-steps", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.0e-6)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype", choices=["bf16", "fp32"], default="bf16",
        help="Training supports BF16 or FP32. FP16 GradScaler training is not supported.",
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--mask-token", default="<MASK>")
    parser.add_argument("--mask-token-id", type=int, default=None,
                        help="Existing embedding row used as mask; original Qwen3 defaults to 151669.")
    parser.add_argument(
        "--fix-mistral-regex",
        action=argparse.BooleanOptionalAction,
        default=default_fix_mistral_regex,
        help="Enable the Transformers tokenizer regex correction required by Qwen3.5.",
    )
    parser.add_argument("--enable-thinking-template", action="store_true")
    parser.add_argument("--disable-thinking-template", dest="enable_thinking_template", action="store_false")
    parser.set_defaults(enable_thinking_template=True)
    add_d3d_fsdp_args(parser)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.set_defaults(
        persistent_workers=True,
        pin_memory=True,
    )
    return parser.parse_args()
