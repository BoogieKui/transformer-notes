# 第三方材料声明

本仓库是一份**个人学习笔记**。除 `notes/` 下的中文笔记、`code/` 下的代码、以及自绘插图外，仓库内还包含若干第三方材料，**版权均归原作者所有**。这里集中说明来源与许可，便于查阅。

---

## 1. 论文 PDF

`notes/01-*` ~ `notes/06-*/papers/` 及各目录下的论文 PDF，均从 arXiv 下载，仅作个人学习参考。

| 论文 | arXiv | 许可 |
|---|---|---|
| Chain-of-Thought Prompting (Wei et al., 2022) | 2201.11903 | **CC BY 4.0** |
| Direct Preference Optimization (Rafailov et al., 2023) | 2305.18290 | **CC BY 4.0** |
| Extending Context Window via Position Interpolation (Chen et al., 2023) | 2306.15595 | **CC BY 4.0** |
| Introduction to Transformers: an NLP Perspective (Xiao & Zhu, 2023) | 2311.17633 | **CC BY 4.0** |
| RoFormer / RoPE (Su et al., 2021) | 2104.09864 | **CC BY-NC-ND 4.0** |
| 其余 arXiv 论文（Kaplan / Chinchilla / InstructGPT / GPT-3 / BERT / RoBERTa / ELMo / RAG / Toolformer / ViT / FlashAttention / Attention Is All You Need 等） | — | **arXiv.org perpetual, non-exclusive license** |
| GPT-1 (Radford et al., 2018)、GPT-2 (Radford et al., 2019) | 非 arXiv | OpenAI 技术报告，版权归 OpenAI |

每篇论文的原始链接见对应笔记末尾的**参考**小节。arXiv 论文的版权由各作者保留，arXiv 仅获非独占分发权；本仓库转载仅为个人学习用途，如需使用请以 arXiv 原文为准。

---

## 2. 插图

| 来源 | 文件 | 许可 |
|---|---|---|
| Jay Alammar, *The Illustrated Transformer*（[原文](https://jalammar.github.io/illustrated-transformer/)） | `illustrated-logits-label.png`、`illustrated-loss-target.png`、`illustrated-output-head.png` | **CC BY-NC-SA 4.0** |
| 论文原图裁切 | Bahdanau et al. 2014 Fig 1/3、Vaswani et al. 2017 Fig 2、Devlin et al. 2018 Fig 1/2、Radford et al. 2018 Fig 1、Peters et al. 2018 Fig 1、Brown et al. 2020 Fig 1.1/1.3、Wang et al. 2022 Fig 3 | 版权归各原作者，学术引用 |
| 本仓库自绘（matplotlib） | `sigmoid.png`、`softmax.png`、`tanh.png`、`relu.png` | 随本仓库 |

Alammar 的插图按其许可要求署名，署名格式：Alammar, J (2018). *The Illustrated Transformer* [Blog post]. Retrieved from https://jalammar.github.io/illustrated-transformer/

---

## 3. 数据集

| 文件 | 来源 | 许可 |
|---|---|---|
| `data/tinystories.txt` | TinyStories (Eldan & Li, 2023)，Microsoft Research | **CDLA-Sharing-1.0** |
| `data/bpe_merges.json` | 本仓库用上述语料训练 BPE 得到的产物 | 随本仓库 |

---

## 4. 代码

`code/` 下的 `bpe.py`、`mini_gpt.py`、`mini_gpt_bpe.py`、`mini_gpt_real.py` 为本仓库从零实现，用于配套 `notes/` 中的笔记。写作时参考了 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)（**MIT License**）。

---

## 5. 第三方仓库

`neural_networks_and_deep_learning_master/` 是 Michael Nielsen《Neural Networks and Deep Learning》的配套代码仓库副本，**MIT License**，Copyright (c) 2012-2022 Michael Nielsen。完整许可原文见该目录下的 `README.md`。

---

## 6. 本仓库自身

本仓库**未附加 LICENSE**，默认保留所有权利。笔记内容可自由阅读，但未经许可不得复制、改编或再分发。
