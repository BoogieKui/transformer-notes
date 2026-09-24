# Jay Alammar《The Illustrated Transformer》阅读地图

> 2026-08-28 · 本篇不是新笔记，是一张**指引卡**：把 Alammar 的图解和 `01–04` 四篇笔记对上。
>
> **前置知识**：`01-attention.md` ~ `04-transformer-layer.md`。那里讲"为什么这样设计"；Alammar 讲"它长什么样"。本篇只负责把两边缝起来。

---

## 0. 为什么要有这篇：四篇是零件图，缺一张整机图

前四篇按概念**横切**，每篇讲清一个零件：

| 篇 | 零件 | 一句话 |
|---|---|---|
| 01 | encoder–decoder attention | 解码器越过固定向量，每步回头软选源句 |
| 02 | self-attention + Q/K/V + √d_k | 每个词一步直达整句话，用并行换掉 RNN 串行链 |
| 03 | multi-head + 位置编码 | 一份注意力预算不够用、顺序被弄丢了，各补一手 |
| 04 | FFN + 残差/LN + 完整层 + 输出头/解码 | 补非线性、稳训练、拼出 encoder–decoder、让它开口 |

零件齐了，但四篇**谁也没回答**：一条句子从进去到吐出来，中间到底路过哪些站？——这个"整机信号流"的问题，**Jay Alammar 的图解已经替你画好了，而且画得比任何文字复述都好**。所以本篇不再重讲一遍，只做一件事：**把它和你手上的零件图对上。**

> 用芯片的话：前四篇是每个模块的**电路图 + 设计依据**；Alammar 那篇是整颗芯片的**信号流图**。你要的只是"这张信号流图里每个站，对应回哪张电路图"。

---

## 1. 阅读地图

**原文（图文并茂，先读这个）：** Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（遵循 CC BY-NC-SA）

他每一节讲的东西，对应回你笔记的"为什么"：

| Alammar 的章节 | 讲什么 | 回看我的笔记 |
|---|---|---|
| A High-Level Look | encoder/decoder 两摞堆叠、黑线连接 | 04 §14–15（cross-attention / 完整结构） |
| Self-Attention（直觉 → Q/K/V → √d_k） | 每个词 attend 整句、三角色、缩放 | 02 §1–3 |
| The Beast With Many Heads | 多组投影并行跑多个"注意力" | 03 §2 |
| Positional Encoding | 补上顺序 | 03 §7 |
| The Residuals | 残差旁路 | 04 §7 |
| The Decoder Side | masked + cross-attention | 04 §12–14 |
| Final Linear & Softmax | 输出头 | 04 §16 |
| Recap of Training / Loss | 训练目标 | 04 §17 |

> 注意：Alammar 那篇对**输出头机制、交叉熵、greedy/beam 解码**只点到为止——这三块已经展开在 04 §16–18，别指望从图上读出来。

---

## 参考

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)——本篇对照的全部图解来源（遵循 CC BY-NC-SA）
- Vaswani et al., "Attention Is All You Need" (2017)——架构原始出处（[论文 PDF](papers/1706.03762.pdf)）
- 前四篇笔记：`01-attention.md` ~ `04-transformer-layer.md`——每个零件的"为什么"
