"""
bpe_presplit_ab.py —— 预分词到底换来了什么：A/B 对照实验

笔记 `notes/04-karpathy-nanogpt/04-bpe.md` §2 细节 ① 里那几个数，由这个脚本产出。

问题：预分词（把文本切成词、空格粘在词上）是 BPE 里唯一一处"人为规则"。
      它禁止了跨词边界的合并，直觉上会牺牲压缩率。那它换回来什么？

两个变体，除了"要不要预分词"以外完全一样：
  A  预分词 + 按词频统计   —— code/bpe.py 的实现，GPT-2 原版走的就是这条路
  B  纯字符流，不预分词     —— 整段文本是一个符号序列，任意相邻对都能合并

结论（实测）：
  - 压缩率：B 略胜，A 的 token 数多 0.75%  —— 预分词确实牺牲了一点压缩率
  - 统计量：每轮「数相邻对」，A 配对 24,896 次，B 配对 199,999 次  —— A 少 8 倍
            （全量 200 万字符上差距更大：99,636 次 vs 1,999,999 次，少 20 倍）
  - 耗时：这个差距随语料变大而拉开（5 万字符 3.0 倍 -> 40 万字符 5.9 倍）
  - 所以预分词买的不是压缩率，是**统计效率**——它是让 BPE 在大语料上跑得动的那一步

跑一趟约 1 分钟。
"""

import io
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 让 `import bpe` 找到同目录的文件
from bpe import BPETokenizer, _merge, pre_split
from bpe import train as train_A

CORPUS = Path(__file__).resolve().parent.parent / 'data' / 'tinystories.txt'
TEXT = io.open(CORPUS, encoding='utf-8').read()[:2_000_000]

N_MERGES = 500     # A/B 对照用的合并次数
N_TR = 200_000     # A/B 对照：训练段
N_HE = 200_000     # A/B 对照：验证段（A/B 都没见过）


# =====================================================================
# 变体 B：纯字符流。唯一的区别就是不预分词。
# =====================================================================
def train_B(text, n_merges):
    """整段文本当成一个符号序列，任意相邻对都可以合并。"""
    syms = list(text)
    merges = []
    for _ in range(n_merges):
        pairs = Counter(zip(syms, syms[1:]))
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]   # 平局按字母序，保证可复现
        merges.append(best)
        syms = list(_merge(tuple(syms), best))
    return merges


def encode_B(text, merges):
    """B 的编码：按合并优先级从高到低，在整条序列上逐条重放。没有"词"可以缓存。"""
    syms = list(text)
    for a, b in merges:
        c, out, i = a + b, [], 0
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                out.append(c)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        syms = out
    return syms


# =====================================================================
# 一、一眼看穿的例子：为什么"按词统计"能少扫这么多
# =====================================================================
def demo_toy():
    print("=" * 72)
    print("一、一眼看穿的例子：语料 = ' the cat' 重复 20 万次")
    print("=" * 72)

    # 写成 ' the cat'（前导空格）而不是 'the cat '：后者会让词典里多出
    # {'the': 1, ' ': 1} 两条边界残渣，虽然也是真话，但会让人分心。
    toy = ' the cat' * 200_000
    freq = Counter(pre_split(toy))

    raw_ops = len(toy) - 1                                     # 纯字符流：每个位置取一对
    word_ops = sum(len(w) - 1 for w in freq)                   # 按词统计：每个词只数一次

    print(f"  语料 {len(toy):,} 个字符")
    print()
    print(f"  不预分词：输入是 {len(toy):,} 长的字符序列，每轮沿它走一遍")
    print(f"            -> 数一遍相邻对 ~ {raw_ops:,} 次操作")
    print()
    print(f"  预分词后：输入是 {len(freq)} 条记录的词典 {dict(freq)}")
    print(f"            -> 数一遍相邻对 ~ {word_ops} 次操作")
    print()
    print(f"  {raw_ops:,} 次  vs  {word_ops} 次，差 {raw_ops/word_ops:,.0f} 倍")
    print("  差在哪：不预分词时，某个词出现 20 万次，那 20 万个位置每一个都要单独扫；")
    print("          预分词把它折叠成「1 条记录 + 一个数字 200000」——扫一遍，然后乘回去。")


# =====================================================================
# 二、A/B 对照：压缩率、耗时、编码速度
# =====================================================================
def demo_ab():
    print()
    print("=" * 72)
    print(f"二、A/B 对照（训练段 {N_TR:,} 字符，验证段另取 {N_HE:,} 字符，各 {N_MERGES} 次合并）")
    print("=" * 72)

    train = TEXT[:N_TR]
    held = TEXT[N_TR:N_TR + N_HE]

    # 两个变体都用「训练段+验证段」的字符并集作初始词表，
    # 否则验证段里没出现过的字符会让 A 直接 KeyError（bpe.py docstring 里写的那个限制）
    base = sorted(set(train) | set(held))
    freq_tr = freq_of(train)

    n_occ = sum(freq_tr.values())            # 词的个数（含重复）
    n_dis = len(freq_tr)                     # 不同的词
    n_sym = sum(len(w) for w in freq_tr)     # 这些不同的词一共占多少个符号
    n_pair = sum(len(w) - 1 for w in freq_tr)
    print(f"  训练段 {len(train):,} 个字符 -> {n_occ:,} 个词 -> 其中不同的只有 {n_dis:,} 个"
          f"（占词数的 {n_dis/n_occ*100:.1f}%）")
    print(f"  平均每个词重复 {n_occ/n_dis:.1f} 次 —— 这就是「折叠」能省的倍数")
    print()
    print(f"  每轮「数相邻对」的配对次数：")
    print(f"    不预分词  扫 {len(train):,} 个符号 -> {len(train)-1:,} 次配对")
    print(f"    预分词    扫 {n_sym:,} 个符号 -> {n_pair:,} 次配对"
          f"   （少 {(len(train)-1)/n_pair:.1f} 倍）")
    print()

    t0 = time.time()
    _, merges_A = train_A(train, N_MERGES)
    tA = time.time() - t0
    tokA = BPETokenizer(base, merges_A)
    print(f"  A 预分词 + 按词频统计   训练 {tA:5.1f} 秒   词表 {tokA.vocab_size}")

    t0 = time.time()
    merges_B = train_B(train, N_MERGES)
    tB = time.time() - t0
    print(f"  B 纯字符流             训练 {tB:5.1f} 秒   词表 {len(set(train)) + len(merges_B)}"
          f"   （慢 {tB/tA:.1f} 倍）")

    t0 = time.time()
    ids_A_tr, ids_A_he = tokA.encode(train), tokA.encode(held)
    tAe = time.time() - t0
    t0 = time.time()
    ids_B_he = encode_B(held, merges_B)
    tBe = time.time() - t0

    print()
    print(f"  {'':<22}{'训练段 token':>13}{'验证段 token':>13}")
    print("  " + "-" * 48)
    print(f"  {'A 预分词':<22}{len(ids_A_tr):>13,}{len(ids_A_he):>13,}")
    print(f"  {'B 纯字符流':<22}{'—':>13}{len(ids_B_he):>13,}")
    print(f"  {'B 比 A':<22}{'':>13}{(1-len(ids_B_he)/len(ids_A_he))*100:>12.2f}%")
    print(f"  （B 的训练段 token 没算：那要再跑一遍 500 条规则的整段编码，几十秒，不值）")

    print(f"\n  编码 {N_HE:,} 字符的耗时：A {tAe:.1f} 秒，B {tBe:.1f} 秒（A 快 {tBe/max(tAe,1e-9):.0f} 倍）")
    print("  A 快是因为可以按词缓存 —— 同一个词只编码一次，剩下全是查表。这是预分词直接解锁的。")

    assert tokA.decode(ids_A_he) == held, "A 解码回不去"
    assert ''.join(ids_B_he) == held, "B 解码回不去"
    print("\n  解码无损检查：A 通过，B 通过（两个变体都是纯拼接，都能无损还原）")


# =====================================================================
# 三、差距随语料变大而拉开
# =====================================================================
def demo_scaling():
    print()
    print("=" * 72)
    print("三、统计效率的差距随语料变大而拉开（各 300 次合并）")
    print("=" * 72)
    print(f"  {'语料长度':>10}{'不同的词数':>12}{'A 训练':>10}{'B 训练':>10}{'慢多少倍':>10}")
    print("  " + "-" * 52)
    for n in (50_000, 100_000, 200_000, 400_000):
        t = TEXT[:n]
        t0 = time.time(); train_A(t, 300); ta = time.time() - t0
        t0 = time.time(); train_B(t, 300); tb = time.time() - t0
        print(f"  {n:>10,}{len(set(pre_split(t))):>12,}{ta:>9.1f}s{tb:>9.1f}s{tb/ta:>9.1f}x")
    print()
    print("  A 的成本随「不同的词数」涨，B 随「文本长度」涨。")
    print("  文本长度没有上限，不同的词数有 —— 所以差距只会越来越大。")


def freq_of(text):
    return Counter(pre_split(text))


if __name__ == '__main__':
    demo_toy()
    demo_ab()
    demo_scaling()
