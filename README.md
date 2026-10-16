# Transformer 学习笔记

> 以 Stanford CS224N (2025 Winter) 为主线，混合 Illustrated Transformer、Karpathy nanoGPT 等经典资源。

## 学习路线

| 阶段 | 内容 | 状态 | 笔记 |
|------|------|------|------|
| 1 | Word Vectors & Word2Vec | ✅ 完成 | [01-word2vec.md](notes/01-word-vectors/01-word2vec.md) |
| 1 | GloVe & 词向量评估 | ⬜ 待开始 | — |
| 1 | 反向传播复习 & 神经网络基础 | ⬜ 待开始 | —（参考 CS229、Nielsen《NN&DL》） |
| 2 | RNN | ✅ 完成 | [01-rnn-seq2seq.md](notes/02-rnn-seq2seq/01-rnn-seq2seq.md) |
| 2 | LSTM / GRU | ⬜ 支线（可暂缓） | — |
| 2 | Seq2Seq 模型 | ✅ 完成 | [01-rnn-seq2seq.md](notes/02-rnn-seq2seq/01-rnn-seq2seq.md) |
| 3 | Attention 机制 | ✅ 完成 | [01-attention.md](notes/03-attention-transformer/01-attention.md) |
| 3 | Self-Attention & Transformer | ✅ 完成 | [02-self-attention.md](notes/03-attention-transformer/02-self-attention.md) 等 4 篇 |
| 3 | Illustrated Transformer（图解） | ✅ 完成 | [06-illustrated-transformer.md](notes/03-attention-transformer/06-illustrated-transformer.md) |
| 4 | Karpathy nanoGPT（手写实现） | ✅ 完成 | [01-mini-gpt-reading.md](notes/04-karpathy-nanogpt/01-mini-gpt-reading.md) 等 5 篇 |
| 5 | BERT / GPT 预训练 | ✅ 完成 | [01-why-pretraining.md](notes/05-pretraining/01-why-pretraining.md)、[02-bert.md](notes/05-pretraining/02-bert.md)、[03-gpt-in-context.md](notes/05-pretraining/03-gpt-in-context.md)、[04-why-decoder-only.md](notes/05-pretraining/04-why-decoder-only.md) |
| 6 | LLM 关键概念 | ✅ 完成 | [01-llm-concepts.md](notes/06-llm-concepts/01-llm-concepts.md)（一篇概览：从一台续写机器开始 → 规模定律 / 后训练 / 推理 / 效率 / 上下文 / RAG 与 Agent / 多模态 / 量化与蒸馏 / 评估） |

## 资源

- [CS224N 课程主页](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/)
- [D2L 在线书](https://d2l.ai)
- [3Blue1Brown 神经网络系列](https://www.3blue1brown.com/topics/neural-networks)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu)（阶段 4 的搭档：从 tokenization 到训练/scaling/评估的完整课程）
- [Umar Jamil（YouTube，手写 Transformer/GPT）](https://www.youtube.com/@umarjamilai)（阶段 4 的视频搭档）
- [CS229: Machine Learning（Andrew Ng）](https://cs229.stanford.edu)（阶段 1「反向传播复习」补底）
- [Nielsen《Neural Networks and Deep Learning》](http://neuralnetworksanddeeplearning.com/index.html)（阶段 1「反向传播复习」补底）
- [Chollet《Deep Learning with Python》](https://deeplearningwithpython.io/chapters/)（动手实践，留给 code/ 阶段）

## 目录结构

```
notes/              # 学习笔记
  01-word-vectors/  # 词向量
  02-rnn-seq2seq/   # RNN & Seq2Seq
  03-attention-transformer/  # Attention & Transformer
  04-karpathy-nanogpt/      # 手写 GPT
  05-pretraining/   # BERT/GPT 预训练
  06-llm-concepts/  # LLM 关键概念
code/               # 代码实现
```
