"""
bpe.py —— 从零实现的 BPE（Byte Pair Encoding，字节对编码）

为什么单独一个文件：BPE 是**训练一次、用一辈子**的东西。它的产物就是一张
"合并规则表"，存成 JSON 之后，训练和推理就完全分开了。
所以这里只管 tokenizer，不碰模型。模型见 mini_gpt_bpe.py。

背景：BPE 本是 1994 年 Gage 提出的**数据压缩**算法——把出现最多的相邻符号对
合并成一个新符号，重复 N 次，文本就变短了。2016 年 Sennrich 等人把它搬到
NLP 做子词切分，GPT-2 起成为 LLM 标配。它解决的是字符级和词级各自的硬伤：
  - 字符级：词表极小，但 "mouse" 要 5 个 token 才拼得出来，序列太长
  - 词级：一个词 1 个 token，但词表爆炸 + 未登录词（OOV）无解
  - BPE：高频片段（比如 "ing"、" the"）直接进词表，低频词拆成可复用的碎片

和 GPT-2 的一个区别，这里说清楚：GPT-2 的初始词表是 **256 个字节**，所以任何
输入都能编码。这里为了和 mini_gpt_real.py 逐项对照，初始词表用的是**语料里
出现过的 87 个字符**——好处是编号 0~86 和字符级模型完全一致，BPE 只是往后
追加。代价是遇到语料外的字符（比如中文）会 KeyError——想修，把 `base` 换成
`[bytes([i]) for i in range(256)]` 就行，GPT-2 就是这么干的。
"""

import json
import re
from collections import Counter
from pathlib import Path

# =====================================================================
# 0. 语料（和 mini_gpt_real.py 读的是同一份、同一段）
# =====================================================================
CORPUS = Path(__file__).resolve().parent.parent / 'data' / 'tinystories.txt'
N_CHARS = 2_000_000   # 和 mini_gpt_real.py 保持一致，这样两边看的字节完全相同


# =====================================================================
# 1. 预分词：把文本切成"词"，每个词自带前面的那个空格
# =====================================================================
# BPE 不该跨过空格去合并。如果不先切，`e`+` ` 这种跨词边界的对也会被合并，
# 规则表里会塞满 "e "、"s " 这种没意义的碎片。
#
# 关键设计：**把前导空格粘在词上**。`" Once"` 是一个词，`"once"` 是另一个词。
# 这样 BPE 能学到 " the"（带空格）和 "the"（不带）是两个不同的 token——
# 正好对应英文里"词首"和"词中"两种形态。解码时又不需要额外加空格，
# 直接把 token 拼起来就是原文，天然无损。
SPLIT = re.compile(r' ?[^\s]+|\s+')


def pre_split(text):
    return SPLIT.findall(text)


# =====================================================================
# 2. 训练：反复找出"出现次数最多的相邻对"，合并它
# =====================================================================
def train(text, n_merges):
    """返回 (base 字符表, merges 列表)。merges[i] = 第 i 次合并的 (左, 右)。"""
    freq = Counter(pre_split(text))
    # 每个唯一词表示成一个"符号序列"，初始就是它的字符
    parts = {w: tuple(w) for w in freq}
    base = sorted({c for w in freq for c in w})

    merges = []
    for i in range(n_merges):
        # 统计所有相邻对，**按词频加权**——这是快的关键：
        # 2,000,000 个字符里其实只有几万个不同的词，"the" 出现 10 万次
        # 只统计一次、乘以 10 万，而不是老老实实数 10 万遍。
        pairs = Counter()
        for w, f in freq.items():
            s = parts[w]
            for pair in zip(s, s[1:]):
                pairs[pair] += f
        if not pairs:
            break
        # 取频率最高的对；频率相同时按字母序取，保证结果可复现
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]

        merges.append(best)
        # 把所有词里的这个对就地合并掉，下一轮基于合并后的结果统计
        for w in parts:
            parts[w] = _merge(parts[w], best)
    return base, merges


def _merge(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


# =====================================================================
# 3. 编码 / 解码
# =====================================================================
class BPETokenizer:
    def __init__(self, base, merges):
        self.base = list(base)
        self.merges = [tuple(m) for m in merges]
        # 词表 = 87 个字符 + 每次合并产生的一个新 token，编号依次往后排。
        # 所以 stoi['a'] 在字符级模型和这里**是同一个数**，可以直接对照。
        self.itos = self.base + [a + b for a, b in self.merges]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        # 合并优先级：数字越小越先合并。编码时就靠它决定"先合哪个"。
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self._cache = {}                       # 同一个词只编码一次

    @property
    def vocab_size(self):
        return len(self.itos)

    def _encode_word(self, word):
        symbols = list(word)
        while len(symbols) >= 2:
            # 在所有相邻对里，挑优先级最高（rank 最小）的那个合并
            pairs = set(zip(symbols, symbols[1:]))
            best = min(pairs, key=lambda p: self.rank.get(p, float('inf')))
            if best not in self.rank:
                break                          # 剩下的对都没学过，停
            symbols = list(_merge(tuple(symbols), best))
        return symbols

    def encode(self, text):
        ids = []
        for w in pre_split(text):
            if w not in self._cache:
                self._cache[w] = [self.stoi[s] for s in self._encode_word(w)]
            ids.extend(self._cache[w])
        return ids

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(
            {'base': self.base, 'merges': self.merges}, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d['base'], [tuple(m) for m in d['merges']])


# =====================================================================
# 4. 单独跑一遍：训练、看规则、量压缩率
# =====================================================================
if __name__ == '__main__':
    import time

    text = open(CORPUS, encoding='utf-8').read()[:N_CHARS]
    print(f"语料 {len(text):,} 个字符，{len(pre_split(text)):,} 个词，"
          f"{len(set(pre_split(text))):,} 个不同的词")

    N_MERGES = 500
    t0 = time.time()
    base, merges = train(text, N_MERGES)
    print(f"训练 {N_MERGES} 次合并用了 {time.time() - t0:.1f} 秒")

    tok = BPETokenizer(base, merges)
    print(f"词表大小 = {len(base)} 字符 + {len(merges)} 合并 = {tok.vocab_size}")

    print("\n前 25 条合并规则（优先级从高到低）：")
    for i, (a, b) in enumerate(merges[:25]):
        print(f"  {i:3d}. {a!r} + {b!r} -> {a + b!r}")

    ids = tok.encode(text)
    assert tok.decode(ids) == text, "解码回不去，有问题"
    print(f"\n编码结果：{len(text):,} 字符 -> {len(ids):,} 个 token，"
          f"压缩 {len(text) / len(ids):.2f} 倍")
    print(f"往返验证：decode(encode(text)) == text  ✓")

    demo = "Once upon a time, there was a little girl named Lily."
    print(f"\n切分示例：{demo!r}")
    print("  " + " | ".join(repr(tok.itos[i]) for i in tok.encode(demo)))
