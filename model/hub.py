"""Model ids for StreamTN.

The published model is a standard Hugging Face model repository (config,
tokenizer and model.safetensors), so inference is a single
``AutoModelForCausalLM.from_pretrained(model_id)`` call.
"""

from __future__ import annotations

import os

# Training starts from the raw Qwen3-0.6B backbone.
BACKBONE_MODEL_ID = "Qwen/Qwen3-0.6B"

# Published StreamTN weights on the Hugging Face Hub.
# Override with --model or STREAMTN_MODEL_ID for a fork or a local directory.
DEFAULT_MODEL_ID = "starstarcat/StreamTN"


def get_model_id(override: str | None = None) -> str:
    """Inference model: HF repo id or a local HF model directory."""
    return override or os.environ.get("STREAMTN_MODEL_ID") or DEFAULT_MODEL_ID


def get_backbone_id(override: str | None = None) -> str:
    """Training backbone: HF repo id or local directory."""
    return override or os.environ.get("STREAMTN_BACKBONE_ID") or BACKBONE_MODEL_ID
