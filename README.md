<h1 align="center">StreamTN</h1>

<p align="center">
  <b>A Low-Latency Streaming Chinese Text Normalization Model for Streaming TTS in Dialogue Systems</b>
</p>

<p align="center">
  <b>0.6B LM · Dual-Track Streaming · 213 ms First-Packet Delay · Chinese TN Benchmark</b>
</p>

<p align="center">
  English | <a href="README_zh.md">简体中文</a><br>
  <a href="assets/StreamTN_paper.pdf">Paper</a> ·
  <a href="https://supernova-neko.github.io/Stream-TN/">Demo Page</a> ·
  <a href="https://huggingface.co/starstarcat/StreamTN">Hugging Face</a>
</p>

StreamTN is a lightweight Chinese streaming text normalization (TN) model for LLM-centered spoken dialogue systems. It sits between the upstream LLM and the downstream TTS module, converting partial LLM output into TTS-readable Chinese text without waiting for the complete response.

## Overview

![StreamTN architecture](assets/streamtn_architecture.png)

*Figure 1. StreamTN in a cascaded spoken dialogue system (top) and its dual-track streaming architecture built on Qwen3-0.6B (bottom).*

In a cascaded dialogue system, the upstream LLM emits text token by token. Waiting for the full sentence before normalization adds roughly 1.4 s of first-packet delay at a typical upstream rate of 20 tokens/s. StreamTN avoids this with a **dual-track streaming framework** built on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): input tokens and output tokens are processed on two parallel tracks and fused frame by frame, with a fixed delay. Full-parameter fine-tuning removes the need for prompt engineering and reduces hallucinations compared with rule-based systems and prompted general-purpose LLMs.

- **Streaming by design**: with a 4-frame delay, the model reaches Micro-F1 **0.8937** at a first-packet delay (FPD) of **213 ms** on the 1,262-sample benchmark.
- **No prompt engineering**: the normalized text is generated directly by the fine-tuned model.
- **Compact**: a 0.6B-parameter backbone, loaded with a single `AutoModelForCausalLM.from_pretrained` call.
- **Broad coverage**: the benchmark spans numbers, dates, times, units, currency, chemistry, and mathematical expressions.

Developed by ASLP@NPU (Northwestern Polytechnical University) and Shenzhen Pimei Technology Co., Ltd.

## Demo

The [demo page](https://supernova-neko.github.io/Stream-TN/) hosts the material behind the paper, including:

- **Case studies** across text categories, with reference normalization, baseline outputs, and raw / StreamTN-synthesized audio
- **Delay-frame analysis**, relating the delay setting to accuracy and FPD
- **Supplementary experiments**: latency comparison against non-streaming baselines, full evaluation metrics, the LLM baseline prompt, zero-vector and LoRA ablations, and character-level error analysis

## Quickstart

Model weights are downloaded from Hugging Face automatically. Use `--model` or `STREAMTN_MODEL_ID` to select another repository or a local model directory.

### Installation

Python 3.10 and Linux x86_64 are recommended. Run the following commands from the repository root:

```bash
conda create -n streamtn python=3.10 -y
conda activate streamtn
pip install -r requirements.txt
```

For training logs, install WandB after the base requirements:

```bash
pip install wandb
```

The base requirements pin `torch>=2.1,<2.10` for CUDA 12.x drivers. On newer NVIDIA drivers you may remove the upper bound. CPU-only inference works without WandB.

### Inference

```bash
python infer_single.py --text "他今天走了12.5公里"
```

```text
Input : 他今天走了12.5公里
Output: 他今天走了十二点五公里
```

Weights are downloaded automatically from [starstarcat/StreamTN](https://huggingface.co/starstarcat/StreamTN) on first use. To use a fork or a local HF model directory, pass `--model` or set `STREAMTN_MODEL_ID`:

```bash
python infer_single.py --model your-org/your-fork --text "电脑的售价是4899元" --device cuda
```

`--device` accepts `auto` (default), `cpu`, `cuda`, `cuda:0`, and similar values.

### Model weights

| Setting | Default | Override |
| :--- | :--- | :--- |
| Inference model | `starstarcat/StreamTN` | `--model` or `STREAMTN_MODEL_ID` |
| Training backbone | `Qwen/Qwen3-0.6B` | `STREAMTN_BACKBONE_ID` |

Both settings accept a Hugging Face repo id or a local directory containing `config.json`, tokenizer files, and `model.safetensors`.

To publish a checkpoint you trained yourself, `tools/prepare_hf_model.py` converts a training-format `model.safetensors` into a standard HF model directory that can be uploaded to the Hub:

```bash
python tools/prepare_hf_model.py --checkpoint /path/to/training/model.safetensors --output hf_release
```

## Training

Training starts from the Qwen3-0.6B backbone. Provide your own JSONL data (one object with a `messages` list per line; see `train_data/example_data` for the format) and set `data_path` in `config/train_stream.jsonl`, then:

```bash
bash train_stream.sh
```

The launch script uses `CUDA_VISIBLE_DEVICES=0,1` with two processes by default; adjust it to match your GPUs. `resume_from_checkpoint` may point to a Hugging Face repo id or a local HF model directory to start from a prepared checkpoint instead of the raw backbone.

## Evaluation

The paper introduces a Chinese TN benchmark with 1,262 test samples. Character-level Micro-F1 is the primary metric because it accepts pronunciation-equivalent variants. The four tables below reproduce Tables I–IV of the paper; latency decomposition, the LLM baseline prompt, LoRA rank ablation, and error analysis are on the [demo page](https://supernova-neko.github.io/Stream-TN/).

### Main results (Table I)

StreamTN is compared with a rule-based toolkit, a prompted general-purpose LLM, and neural TN baselines:

| Method | Micro-F1 | Micro-P |
| :--- | ---: | ---: |
| WeTextProcessing | 0.7853 | 0.7586 |
| Qwen3-0.6B (prompted) | 0.4872 | 0.5282 |
| FlatTN | 0.7569 ± 0.0019 | 0.7390 ± 0.0017 |
| BiLSTM | 0.8941 ± 0.0015 | 0.8677 ± 0.0012 |
| **StreamTN** | 0.8937 ± 0.0009 | 0.8931 ± 0.0022 |

StreamTN uses the 4-frame streaming setting with a 213 ms FPD and achieves the best Micro-P. BiLSTM is marginally higher on Micro-F1 but is non-streaming and must wait for the complete input.

### Per-category results (Table II)

Micro-averaged precision, recall and F1 of StreamTN across 14 TN categories:

| TN Type | Micro-P | Micro-R | Micro-F1 |
| :--- | ---: | ---: | ---: |
| Integers and Decimals | 0.959 ± 0.006 | 0.963 ± 0.004 | 0.961 ± 0.004 |
| Ordinal Numbers | 0.923 ± 0.010 | 0.929 ± 0.005 | 0.926 ± 0.007 |
| Phone Numbers | 0.919 ± 0.008 | 0.917 ± 0.008 | 0.918 ± 0.008 |
| Years and Dates | 0.949 ± 0.009 | 0.957 ± 0.009 | 0.953 ± 0.009 |
| Time | 0.956 ± 0.002 | 0.978 ± 0.003 | 0.967 ± 0.001 |
| Fractions and Percentages | 0.964 ± 0.009 | 0.947 ± 0.006 | 0.955 ± 0.007 |
| Military Passwords | 0.925 ± 0.003 | 0.924 ± 0.004 | 0.925 ± 0.003 |
| Capital Amounts | 0.970 ± 0.004 | 0.979 ± 0.001 | 0.975 ± 0.002 |
| Amount Reading | 0.927 ± 0.006 | 0.919 ± 0.011 | 0.923 ± 0.007 |
| Physical Units | 0.976 ± 0.003 | 0.978 ± 0.003 | 0.977 ± 0.003 |
| Chemical Formulas | 0.785 ± 0.008 | 0.797 ± 0.006 | 0.791 ± 0.006 |
| Daily Chemicals | 0.913 ± 0.008 | 0.884 ± 0.009 | 0.898 ± 0.007 |
| Simple Math Equations | 0.956 ± 0.005 | 0.943 ± 0.005 | 0.950 ± 0.005 |
| Complex Math Equations | 0.749 ± 0.014 | 0.751 ± 0.007 | 0.750 ± 0.009 |

Regular, frequently observed categories (physical units, capital amounts, time) exceed 0.96 Micro-F1. Complex mathematical equations and chemical formulas remain the most challenging at 0.750 and 0.791. The small standard deviations indicate consistent results across training seeds.

### Ablation (Table III)

| Method | Micro-F1 | Micro-P |
| :--- | ---: | ---: |
| **StreamTN** | 0.8937 ± 0.0009 | 0.8931 ± 0.0022 |
| w/ LoRA Fine-tuning | 0.6335 ± 0.0012 | 0.6250 ± 0.0015 |
| w/ System Prompt | 0.8852 ± 0.0018 | 0.8824 ± 0.0021 |

Full-parameter fine-tuning is important for the precise transformations required by TN, and StreamTN generates structured normalized text without carefully engineered prompts.

### Delay frames (Table IV)

FPD denotes first-packet delay in milliseconds; `(main)` marks the configuration used in the main experiments.

| Strategy | Delay frame | Micro-F1 | Micro-P | FPD (ms) |
| :--- | :--- | ---: | ---: | ---: |
| Non-Streaming | — | 0.9639 ± 0.0011 | 0.9620 ± 0.0012 | — |
| Streaming | 1 frame | 0.7030 ± 0.0010 | 0.7135 ± 0.0018 | 75 |
| Streaming | 2 frames | 0.7952 ± 0.0015 | 0.8062 ± 0.0026 | 121 |
| Streaming | 4 frames (main) | 0.8937 ± 0.0009 | 0.8931 ± 0.0022 | 213 |
| Streaming | 8 frames | 0.9072 ± 0.0011 | 0.9093 ± 0.0020 | 394 |
| Streaming | 16 frames | 0.9239 ± 0.0008 | 0.9303 ± 0.0015 | 756 |

More delay frames give the model more context and improve normalization, at the cost of a longer FPD (75 ms → 756 ms). The 4-frame setting is the practical trade-off, reaching Micro-F1 0.8937 at 213 ms. The non-streaming variant has the highest accuracy but must wait for the complete input.

## Citation

```bibtex
@misc{li2026streamtn,
  title  = {StreamTN: A Low-Latency Streaming Chinese Text Normalization Model for Streaming TTS in Dialogue Systems},
  author = {Li, Wenhao and Liang, Jinrui and Zhang, Haoyu and Hu, Jingbin and Ren, Xiaming and Xie, Hanke and Chen, Huakang and Wang, Chengyou and Guo, Dake and Ma, Linhan and Feng, Su and Liu, Houdun and Chen, Yunxiang and Xie, Lei},
  year   = {2026},
  url    = {https://supernova-neko.github.io/Stream-TN/}
}
```
