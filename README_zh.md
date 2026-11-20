<h1 align="center">StreamTN</h1>

<p align="center">
  <b>面向流式 TTS 对话系统的低延迟中文流式文本正则化模型</b>
</p>

<p align="center">
  <b>0.6B LM · 双轨流式 · 213 ms 首包延迟 · 中文 TN 评测基准</b>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文<br>
  <a href="assets/StreamTN_paper.pdf">论文</a> ·
  <a href="https://supernova-neko.github.io/Stream-TN/">Demo 页面</a> ·
  <a href="https://huggingface.co/starstarcat/StreamTN">Hugging Face</a>
</p>

StreamTN 是面向以大语言模型为中心的口语对话系统的轻量级中文流式文本正则化（Text Normalization, TN）模型。它位于上游 LLM 与下游 TTS 模块之间，将 LLM 逐步输出的部分文本转换为可供 TTS 合成的中文文本，无需等待整句生成完毕。

## 概述

![StreamTN 整体结构](assets/streamtn_architecture.png)

*图 1：StreamTN 在级联口语对话系统中的位置（上），以及基于 Qwen3-0.6B 的双轨流式架构（下）。*

在级联式对话系统中，上游 LLM 逐 token 生成文本。若等整句生成后再做正则化，按典型上游速度 20 tokens/s 计算，首包延迟约为 1.4 秒。StreamTN 基于 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，采用**双轨流式框架**：输入 token 与输出 token 在两条并行轨道上逐帧处理并融合，通过固定延迟帧数控制时延与准确率。全参数微调无需提示词工程，相比规则系统与通用 LLM 提示方案，可减少幻觉并提升正则化效果。

- **原生流式**：4 帧延迟配置下，在 1,262 条测试样本上取得 Micro-F1 **0.8937**，首包延迟（FPD）**213 ms**。
- **无需提示词**：归一化文本由微调后的模型直接生成。
- **模型轻量**：0.6B 参数量底座，一次 `AutoModelForCausalLM.from_pretrained` 即可加载。
- **覆盖广泛**：评测涵盖数字、日期、时间、计量单位、金额、化学与数学表达式等场景。

由西北工业大学 ASLP@NPU 与 Shenzhen Pimei Technology Co., Ltd. 共同完成。

## 在线 Demo

[Demo 页面](https://supernova-neko.github.io/Stream-TN/) 提供论文相关材料：

- **样例展示**：各类文本的原始输入、参考归一化结果、基线输出，以及原始音频与 StreamTN 合成音频
- **延迟帧分析**：不同延迟配置与准确率、FPD 的关系
- **补充实验**：与非流式基线的延迟对比、完整评测指标、LLM 基线提示词、零向量消融、LoRA 消融、字符级错误分析

## 快速开始

权重自动从 Hugging Face 下载。可用 `--model` 或 `STREAMTN_MODEL_ID` 选择其他仓库或本地模型目录。

### 安装

推荐 Python 3.10 与 Linux x86_64。以下命令在仓库根目录执行：

```bash
conda create -n streamtn python=3.10 -y
conda activate streamtn
pip install -r requirements.txt
```

如需训练日志，在基础依赖之后安装 WandB：

```bash
pip install wandb
```

基础依赖将 PyTorch 固定为 `torch>=2.1,<2.10`，以兼容 CUDA 12.x 驱动；若使用更新的 NVIDIA 驱动，可去掉该上界。仅做 CPU 推理无需 WandB。

### 推理

```bash
python infer_single.py --text "他今天走了12.5公里"
```

```text
Input : 他今天走了12.5公里
Output: 他今天走了十二点五公里
```

权重首次使用时自动从 [starstarcat/StreamTN](https://huggingface.co/starstarcat/StreamTN) 下载。若要使用自己的 fork 或本地 HF 模型目录，可通过 `--model` 或环境变量 `STREAMTN_MODEL_ID` 覆盖：

```bash
python infer_single.py --model your-org/your-fork --text "电脑的售价是4899元" --device cuda
```

`--device` 支持 `auto`（默认）、`cpu`、`cuda`、`cuda:0` 等取值。

### 模型权重

| 配置 | 默认值 | 覆盖方式 |
| :--- | :--- | :--- |
| 推理模型 | `starstarcat/StreamTN` | `--model` 或 `STREAMTN_MODEL_ID` |
| 训练底座 | `Qwen/Qwen3-0.6B` | `STREAMTN_BACKBONE_ID` |

两者均可填 Hugging Face repo id，或包含 `config.json`、tokenizer 文件与 `model.safetensors` 的本地目录。

若要发布自己训练得到的权重，`tools/prepare_hf_model.py` 可将训练格式的 `model.safetensors` 转换为标准 HF 模型目录，便于上传到 Hub：

```bash
python tools/prepare_hf_model.py --checkpoint /path/to/training/model.safetensors --output hf_release
```

## 训练

训练从 Qwen3-0.6B 底座开始。准备自己的 JSONL 数据（每行一个含 `messages` 列表的对象，格式见 `train_data/example_data`），在 `config/train_stream.jsonl` 中设置 `data_path`，然后运行：

```bash
bash train_stream.sh
```

启动脚本默认使用 `CUDA_VISIBLE_DEVICES=0,1` 与两个进程，请按实际 GPU 数量调整。`resume_from_checkpoint` 可填 Hugging Face repo id 或本地 HF 模型目录，从已准备好的权重而非原始底座开始训练。

## 评测

论文提出了包含 1,262 条测试样本的中文 TN 基准。由于 TTS 场景允许发音等价的不同写法，主指标采用字符级 Micro-F1。以下四张表复现论文 Table I–IV；延迟拆解、LLM 基线提示词、LoRA rank 消融与错误分析见 [Demo 页面](https://supernova-neko.github.io/Stream-TN/)。

### 主实验结果（Table I）

StreamTN 与规则工具、通用 LLM 提示方案及神经 TN 基线对比：

| 方法 | Micro-F1 | Micro-P |
| :--- | ---: | ---: |
| WeTextProcessing | 0.7853 | 0.7586 |
| Qwen3-0.6B（提示） | 0.4872 | 0.5282 |
| FlatTN | 0.7569 ± 0.0019 | 0.7390 ± 0.0017 |
| BiLSTM | 0.8941 ± 0.0015 | 0.8677 ± 0.0012 |
| **StreamTN** | 0.8937 ± 0.0009 | 0.8931 ± 0.0022 |

StreamTN 采用 4 帧流式配置，首包延迟 213 ms，Micro-P 为对比系统中最高；BiLSTM 的 Micro-F1 略高，但属于非流式方法，必须等完整输入到达才能输出。

### 分类别结果（Table II）

StreamTN 在 14 类 TN 场景上的微平均精度、召回与 F1：

| 类别 | Micro-P | Micro-R | Micro-F1 |
| :--- | ---: | ---: | ---: |
| 整数与小数 | 0.959 ± 0.006 | 0.963 ± 0.004 | 0.961 ± 0.004 |
| 序数词 | 0.923 ± 0.010 | 0.929 ± 0.005 | 0.926 ± 0.007 |
| 电话号码 | 0.919 ± 0.008 | 0.917 ± 0.008 | 0.918 ± 0.008 |
| 年份与日期 | 0.949 ± 0.009 | 0.957 ± 0.009 | 0.953 ± 0.009 |
| 时间 | 0.956 ± 0.002 | 0.978 ± 0.003 | 0.967 ± 0.001 |
| 分数与百分数 | 0.964 ± 0.009 | 0.947 ± 0.006 | 0.955 ± 0.007 |
| 军事口令 | 0.925 ± 0.003 | 0.924 ± 0.004 | 0.925 ± 0.003 |
| 大写金额 | 0.970 ± 0.004 | 0.979 ± 0.001 | 0.975 ± 0.002 |
| 金额读法 | 0.927 ± 0.006 | 0.919 ± 0.011 | 0.923 ± 0.007 |
| 物理单位 | 0.976 ± 0.003 | 0.978 ± 0.003 | 0.977 ± 0.003 |
| 化学式与方程 | 0.785 ± 0.008 | 0.797 ± 0.006 | 0.791 ± 0.006 |
| 日常化学物 | 0.913 ± 0.008 | 0.884 ± 0.009 | 0.898 ± 0.007 |
| 简单数学表达式 | 0.956 ± 0.005 | 0.943 ± 0.005 | 0.950 ± 0.005 |
| 复杂数学表达式 | 0.749 ± 0.014 | 0.751 ± 0.007 | 0.750 ± 0.009 |

规律性强、出现频繁的类别（物理单位、大写金额、时间）Micro-F1 超过 0.96；复杂数学表达式与化学式最具挑战，分别为 0.750 与 0.791。较小的标准差说明不同训练种子下结果稳定。

### 消融实验（Table III）

| 方法 | Micro-F1 | Micro-P |
| :--- | ---: | ---: |
| **StreamTN** | 0.8937 ± 0.0009 | 0.8931 ± 0.0022 |
| 改用 LoRA 微调 | 0.6335 ± 0.0012 | 0.6250 ± 0.0015 |
| 加入 System Prompt | 0.8852 ± 0.0018 | 0.8824 ± 0.0021 |

全参数微调对 TN 所需的精确变换很关键；StreamTN 无需精心设计的提示词即可生成结构化归一化文本。

### 延迟帧分析（Table IV）

FPD 为首包延迟（毫秒）；`(主实验)` 表示主实验采用的配置。

| 策略 | 延迟帧数 | Micro-F1 | Micro-P | FPD (ms) |
| :--- | :--- | ---: | ---: | ---: |
| 非流式 | — | 0.9639 ± 0.0011 | 0.9620 ± 0.0012 | — |
| 流式 | 1 帧 | 0.7030 ± 0.0010 | 0.7135 ± 0.0018 | 75 |
| 流式 | 2 帧 | 0.7952 ± 0.0015 | 0.8062 ± 0.0026 | 121 |
| 流式 | 4 帧（主实验） | 0.8937 ± 0.0009 | 0.8931 ± 0.0022 | 213 |
| 流式 | 8 帧 | 0.9072 ± 0.0011 | 0.9093 ± 0.0020 | 394 |
| 流式 | 16 帧 | 0.9239 ± 0.0008 | 0.9303 ± 0.0015 | 756 |

延迟帧数越多，模型可利用的上下文越充分、归一化效果越好，但 FPD 从 75 ms 增至 756 ms。4 帧配置是准确率与响应速度的折中，Micro-F1 0.8937、FPD 213 ms；非流式版本准确率最高，但必须等待完整输入。

## 引用

```bibtex
@misc{li2026streamtn,
  title  = {StreamTN: A Low-Latency Streaming Chinese Text Normalization Model for Streaming TTS in Dialogue Systems},
  author = {Li, Wenhao and Liang, Jinrui and Zhang, Haoyu and Hu, Jingbin and Ren, Xiaming and Xie, Hanke and Chen, Huakang and Wang, Chengyou and Guo, Dake and Ma, Linhan and Feng, Su and Liu, Houdun and Chen, Yunxiang and Xie, Lei},
  year   = {2026},
  url    = {https://supernova-neko.github.io/Stream-TN/}
}
```
