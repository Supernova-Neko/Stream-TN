#!/usr/bin/env python3
"""Convert a training checkpoint into a standard Hugging Face model directory.

Training saves ``stream_tn_model.state_dict()``: keys have an extra ``model.``
prefix and ``lm_head.weight`` is absent because embeddings are tied. This
script strips the prefix, restores ``lm_head.weight`` from the embedding table,
optionally casts to bfloat16, and adds config/tokenizer files from the
backbone. The output directory can then be loaded directly with
``AutoModelForCausalLM.from_pretrained(output)`` or uploaded to the Hub.

Example:
    python tools/prepare_hf_model.py \
        --checkpoint /path/to/exp/.../model.safetensors \
        --output hf_model
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.hub import BACKBONE_MODEL_ID  # noqa: E402

NON_WEIGHT_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def copy_non_weight_files(backbone: str, output: Path) -> None:
    backbone_dir = Path(backbone).expanduser()
    if backbone_dir.is_dir():
        for name in NON_WEIGHT_FILES:
            source = backbone_dir / name
            if source.is_file():
                shutil.copy2(source, output / name)
            else:
                print(f"Warning: missing {source}")
        return

    from huggingface_hub import hf_hub_download

    for name in NON_WEIGHT_FILES:
        try:
            cached = hf_hub_download(repo_id=backbone, filename=name)
        except Exception as exc:
            print(f"Warning: could not download {name} from {backbone}: {exc}")
            continue
        shutil.copy2(cached, output / name)


def convert_state_dict(checkpoint: Path, dtype: torch.dtype) -> dict:
    state = load_file(str(checkpoint))
    converted = {}
    for key, tensor in state.items():
        new_key = key.removeprefix("model.")
        value = tensor.to(dtype) if tensor.is_floating_point() else tensor
        converted[new_key] = value.contiguous()

    if "model.embed_tokens.weight" not in converted:
        raise KeyError("model.embed_tokens.weight not found in the checkpoint")
    if "lm_head.weight" not in converted:
        converted["lm_head.weight"] = converted["model.embed_tokens.weight"].clone()
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Training-format model.safetensors")
    parser.add_argument("--output", default="hf_model", help="Output HF model directory")
    parser.add_argument(
        "--backbone",
        default=BACKBONE_MODEL_ID,
        help="Backbone HF repo id or local directory supplying config/tokenizer files",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
        help="Storage dtype (bfloat16 matches the backbone and halves the size)",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    converted = convert_state_dict(checkpoint, dtype)
    save_file(converted, str(output / "model.safetensors"), metadata={"format": "pt"})
    copy_non_weight_files(args.backbone, output)

    print(f"Wrote {len(converted)} tensors to {output / 'model.safetensors'}")
    print(f"HF model directory: {output.resolve()}")
    print("Load with:  python infer_single.py --model " + str(output))
    print("Upload with: huggingface-cli upload <repo_id> " + str(output) + " .")


if __name__ == "__main__":
    main()
