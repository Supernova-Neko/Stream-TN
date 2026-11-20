#!/usr/bin/env python3
"""Normalize one sentence with StreamTN.

The model is a standard Hugging Face model directory/repo and is loaded with
a single ``AutoModelForCausalLM.from_pretrained`` call.

Examples:
    python infer_single.py --text "他今天走了12.5公里" --model hf_model
    python infer_single.py --text "..." --model your-org/your-repo
"""

import argparse

import torch

from model.hub import get_model_id
from model.model import stream_tn_model


def load_model(model_id: str, device: torch.device):
    print(f"Model:     {model_id}")
    model = stream_tn_model(model_id=model_id).to(device)
    model.eval()
    return model


def extract_normalized_text(decoded_text: str) -> str:
    """Decoded output uses a two-line convention; keep a safe fallback."""
    lines = decoded_text.splitlines()
    if len(lines) >= 2:
        return lines[1].strip()
    return decoded_text.strip()


def infer_single_text(model, raw_text: str, device: torch.device):
    if not raw_text or not raw_text.strip():
        return "", 0.0

    user_text = f"<|im_start|>user\n{raw_text.strip()}<|im_end|>\n"
    token_ids = model.tokenizer(
        user_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    with torch.no_grad():
        result_tokens, cost_time = model.generate(token_ids)

    if not result_tokens:
        return "", float(cost_time)

    result_tensor = torch.stack(result_tokens, dim=1)
    decoded = model.tokenizer.decode(result_tensor[0], skip_special_tokens=True)
    return extract_normalized_text(decoded), float(cost_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamTN single-sentence inference")
    parser.add_argument(
        "--text",
        type=str,
        default="他今天走了12.5公里，这是他每天的锻炼计划",
        help="Sentence to normalize.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Hugging Face repo id or local HF model directory (e.g. hf_model). "
        "Defaults to the repo id in model/hub.py or STREAMTN_MODEL_ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    args = parser.parse_args()

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Device:    {device}")

    model = load_model(get_model_id(args.model), device)
    output, latency = infer_single_text(model, args.text, device)

    print(f"Input : {args.text}")
    print(f"Output: {output}")
    print(f"First-token latency: {latency * 1000:.1f} ms")


if __name__ == "__main__":
    main()
