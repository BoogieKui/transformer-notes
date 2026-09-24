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
| 4 | Karpathy nanoGPT（手写实现） | 🔄 进行中 | —（参考 CS336、Umar Jamil） |
| 5 | BERT / GPT 预训练 | ⬜ 待开始 | — |
| 6 | 《Attention Is All You Need》 | ⬜ 待开始 | — |

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
  06-paper-reading/ # 论文阅读
code/               # 代码实现
```
