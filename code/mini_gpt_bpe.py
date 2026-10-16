"""
mini_gpt_bpe.py —— 同一台 GPT、同一份语料，只把分词器从字符级换成 BPE

和 mini_gpt_real.py 的关系：**语料、架构、超参、优化器逐字相同，只有「文本怎么
变成 token」不一样。** 这样两组实验的差别只能归因于分词器。

唯一的例外是 `generate` 多了一个可选的 `eos_id`（见笔记 04-bpe 第 8 节）：默认 `None`，
行为一模一样，不影响这组对照；传了才启用「模型自己说完了就停」。

一个文件 = 一台整机。对应笔记：
  - notes/03-attention-transformer/03-multihead-position.md  → Q/K/V、multi-head、位置编码
  - notes/03-attention-transformer/04-transformer-layer.md   → FFN、残差、Pre-LN、输出头、训练、解码
  - notes/04-karpathy-nanogpt/03-real-data.md                → 换真语料这一步发生了什么
  - notes/04-karpathy-nanogpt/04-bpe.md                      → 换分词器这一步发生了什么

设计原则（"设计是被挣来的"）：每个零件出现，都是因为上一个零件不够用。
所以这个文件从上到下按「问题 → 解法」铺开，可以顺着读。

还没做（留作下一步）：dropout、更大的模型。
weight tying 不做：理由见 notes/03-attention-transformer/04-transformer-layer.md §16.5。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = 'cpu'  # 最小骨架先用 CPU，跑一趟约 1 分钟；以后想加速一行换成 'mps'（Apple Silicon）

# =====================================================================
# 0. 数据 + BPE 分词器（和 mini_gpt_real.py 唯一实质不同的地方）
# =====================================================================
# 字符级：一个字符 = 一个 token，于是 "mouse" 要 5 个 token 才拼得出来。
# BPE   ：把语料里最高频的相邻字符对反复合并成新 token，" the" 于是变成 1 个。
#         同一段文本的 token 数降到约 1/3，block_size 能装下的内容就多了约 3 倍。
# 分词器本身（训练 + 编解码）在 bpe.py，和模型分开，因为它训练一次、用一辈子。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 让 `import bpe` 找到同目录的文件
from bpe import BPETokenizer, train, pre_split

CORPUS  = Path(__file__).resolve().parent.parent / 'data' / 'tinystories.txt'
N_CHARS = 2_000_000      # 和 mini_gpt_real.py 取同一段，保证两边看的字节完全相同
DATA = open(CORPUS, encoding='utf-8').read()[:N_CHARS]

# 训练 500 次合并要十几秒，所以结果缓存成 JSON：第一次训练并存盘，之后直接加载。
# 缓存里必须核对语料和合并次数——改了 N_CHARS 或 N_MERGES，旧缓存就作废，重训。
N_MERGES = 500
CACHE = Path(__file__).resolve().parent.parent / 'data' / 'bpe_merges.json'
if CACHE.exists():
    tok = BPETokenizer.load(CACHE)
    if tok.base != sorted(set(DATA)) or len(tok.merges) != N_MERGES:
        print("缓存和当前设置对不上，重新训练 BPE")
        tok = None
else:
    tok = None
if tok is None:
    base, merges = train(DATA, N_MERGES)
    tok = BPETokenizer(base, merges)
    tok.save(CACHE)

vocab_size = tok.vocab_size   # 87 个字符 + 500 次合并 = 587
encode, decode = tok.encode, tok.decode

data = torch.tensor(encode(DATA), dtype=torch.long)

# 简单 90/10 划分：train 用来学，val 用来暴露"过拟合"
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"词表大小 vocab_size = {vocab_size}")
print(f"数据长度 = {len(data)} 个 token，理论初始 loss ≈ ln({vocab_size}) = {math.log(vocab_size):.3f}")

# =====================================================================
# 1. 超参数
# =====================================================================
# 刻意让每个数字在全文件里只用一次，这样读代码时"看到数字就知道是哪个维度"：
#   B = batch_size = 32   T = block_size = 50   C = n_embd = 96   V = vocab_size = 587
#   nh = n_head = 6       hd = head_dim = 16    L = n_layer = 3
# （早期版本里 n_embd 和 block_size 都是 64，导致 (B,T,C) 里 T 和 C 分不清，已改开。）
n_embd = 96        # C：每个 token 的隐向量维度
n_head = 6         # 注意力头数（head_dim = n_embd / n_head = 16）
n_layer = 3        # L：Transformer 层数
block_size = 50    # T：上下文长度，一次最多看 50 个 token
batch_size = 32    # B：每个 step 采多少条样本
max_iters = 1500   # 训练步数（CPU 上约 1 分钟），和笔记 03 里字符级那组同口径
lr = 3e-4          # AdamW 学习率
eval_interval = 150


# =====================================================================
# 2. 零件：因果自注意力（笔记 03 第 3、4 节）
# =====================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # 一次线性投影同时得到 Q、K、V（它们都来自同一个 x）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)  # 多头的输出再拼回原维度
        # 因果 mask：位置 i 只能"看"到位置 <= i（不能偷看未来，笔记 04 第 13 节）
        self.register_buffer(
            'mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(n_embd, dim=2)               # 各 (B, T, C)
        # 拆成多个头，方便每个头在更小的子空间里各自算注意力
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 注意力核心：softmax( Q K^T / sqrt(d) ) V
        att = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)      # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v                                                    # (B, nh, T, hd)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# =====================================================================
# 3. 零件：前馈网络 FFN（笔记 04 第 1~5 节）
# =====================================================================
class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # 中间放大 4 倍 + 一个非线性，再压回来。attention 只会"聚信息"，
        # 不会"加工信息"，这里给每个位置单独加工的机会。
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


# =====================================================================
# 4. 一层 = Pre-LN + 注意力 + Pre-LN + FFN，各带残差（笔记 04 第 7、9 节）
# =====================================================================
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        # Pre-LN（现代做法）：先归一化再进子层，最后加残差。比 Post-LN 训练更稳。
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# =====================================================================
# 5. 整机：GPT（笔记 04 第 10、15 节）
# =====================================================================
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)          # token 嵌入表
        self.wpe = nn.Embedding(block_size, n_embd)          # 位置嵌入表（笔记 03 第 5 节）
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)                     # 最后一层归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)         # 输出头：隐向量 → 词表概率

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)                    # (B, T, C)：词义 + 位置
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                             # (B, T, vocab)

        loss = None
        if targets is not None:
            # 训练目标（笔记 04 第 17 节）：让模型对"下一个词"的预测贴近真实的那一个词
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, eos_id=None):
        # 解码（笔记 04 第 18 节）：一个 token 一个 token 地采样补到序列末尾
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]            # 只保留最后 block_size 个 token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature         # 只看最后一个位置的预测
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            # eos_id 默认 None = 行为一字不变（和 mini_gpt_real.py 保持一致，
            # 这样笔记 04-bpe 第 7 节那组对照才成立）。传了才启用「模型自己说完了就停」。
            # 注意 EOS 是采样出来的概率，不是保证，所以上面那层预算不能省。
            if eos_id is not None and next_token.item() == eos_id:
                break
        return idx


# =====================================================================
# 6. 采样一个 batch：随机位置切一段，x 是上文，y 是"下一位"（错开一格）
# =====================================================================
def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        xb, yb = get_batch(split)
        _, loss = model(xb, yb)
        out[split] = loss.item()
    model.train()
    return out


# =====================================================================
# 7. 训练循环
# =====================================================================
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
print(f"\n模型参数量 = {n_params:,}")

for step in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        est = estimate_loss()
        print(f"step {step:4d} | train loss {est['train']:.4f} | val loss {est['val']:.4f}")

# =====================================================================
# 8. 生成演示
# =====================================================================
context = torch.tensor([encode('Once upon a time')], dtype=torch.long, device=device)
EOS_ID = tok.stoi['<|endoftext|>']        # 209，见笔记 04-bpe 第 3 节

print("\n--- 一、不设 eos_id（默认）：讲完一篇会接着讲下一篇 ---")
print(decode(model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()))

print("\n--- 二、设 eos_id=209：它说完了就停 ---")
out = model.generate(context, max_new_tokens=500, temperature=0.8, eos_id=EOS_ID)[0].tolist()
print(decode(out))
print(f"\n（生成了 {len(out) - context.shape[1]} 个 token 后停下，预算本来是 500）")
