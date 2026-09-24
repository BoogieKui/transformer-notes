# Python / PyTorch 语法速查

> 只覆盖 `code/mini_gpt.py` 里实际用到的语法，按"在哪出现"组织，方便边读代码边查。

---

## 0. 一句话核心

**写 PyTorch 就是在写前向计算。** 你写出来的每一行运算，框架都会悄悄记下来、连成一张计算图；当你调用 `loss.backward()` 时，框架自动按这张图反向算出每个参数的梯度（也就是你在 CS224N 学的反向传播），你不需要手写任何 dL/dW。

所以整个文件里，**大部分代码是"前向怎么做"，只有最后训练循环里的三行（`zero_grad` / `backward` / `step`）是"反向怎么更新"**。

---

## 1. Python 部分

### 1.1 `import` —— 引入库

```python
import torch                      # 引入整个 torch 库
import torch.nn as nn             # 引入子模块，并起别名 nn
import torch.nn.functional as F   # 起别名 F，存各种无参数的函数
```

引入一个库/模块，之后就能用里面的东西。`as xx` 是起别名，`nn.Linear` 就是"`torch.nn` 里的 `Linear`"。

### 1.2 变量与赋值 —— 没有类型声明

```python
n_embd = 96        # 整数
lr = 3e-4          # 浮点数
chars = [...]      # 列表
```

Python 是**动态类型**：变量不需要声明类型，直接赋值，类型由值决定，之后也能重新赋成别的类型。

### 1.3 字符串

```python
s = "hello"                        # 单/双引号 = 普通字符串
DATA = """第一行
第二行"""                          # 三引号 = 多行字符串，可跨行
name = f"词表大小 = {vocab_size}"   # f-string：{ } 里放变量，自动替换成它的值
```

### 1.4 列表 `list` 与列表推导式

```python
xs = [1, 2, 3]                    # 列表：有序、可重复
xs.append(4)                       # 加元素
len(xs)                            # 长度 = 4

# 列表推导式：用一行 for 循环生成一个列表
squares = [i * i for i in range(4)]      # [0, 1, 4, 9]
```

列表推导式是 Python 最常见的简写：`[表达式 for 变量 in 可迭代对象]`，等价于"循环一遍，每轮算出表达式，收集成列表"。

### 1.5 字典 `dict` —— 键值对映射

```python
stoi = {'a': 0, 'b': 1, 'c': 2}   # 字典：{键: 值}
stoi['a']                          # 查键 → 0
```

字典是"查表"用的结构：给一个键（key），拿回对应的值（value）。`mini_gpt.py` 里的 `stoi`/`itos` 就是两张互相反向的字典——`stoi` = **str**ing **t**o **i**nt，字符查编号；`itos` = **i**nt **t**o **st**ring，编号查字符。

### 1.6 函数 `def` 与 `lambda`

```python
# 有名函数
def add(a, b):
    return a + b

# lambda：一次性匿名函数，等价于上面那句，只是没名字
add = lambda a, b: a + b
```

`lambda 参数: 表达式` 是单行匿名函数，适合很短的小函数。文件里 `encode`/`decode` 就是用 `lambda` 写的。

### 1.7 类 `class` 与 `self`

```python
class GPT(nn.Module):                        # 括号①：父类
    def __init__(self, vocab_size, n_layer):  # 括号②：形参（比调用多一个 self）
        super().__init__()
        self.vocab_size = vocab_size          # self.xxx = "这个实例自己的属性"
        self.n_layer = n_layer

    def forward(self, idx):                   # 普通方法
        ...

model = GPT(35, 3)                            # 括号③：实参（不传 self）
```

`class` 定义一类对象；`self` 代指"当前这个实例"，`self.xxx` 表示这个实例身上的属性。`__init__` 是构造方法（constructor），`GPT(35, 3)` 这样创建实例时会自动执行它。（真实签名有 5 个参数，这里只留两个好读。）

**三个括号，三个意思：**

| | 括号里是什么 | 什么时候执行 | 含义 |
|---|---|---|---|
| ① `class GPT(nn.Module)` | **父类**（base class） | 定义类的那一刻 | "GPT 是一种 nn.Module" —— 继承 |
| ② `def __init__(self, …)` | 形参 | —— | 声明"造一台 GPT 要哪些原料" |
| ③ `GPT(35, 3)` | 实参 | 创建实例的那一刻 | 真把原料递进去 |

**括号①里不是参数**，那是 Python 的固定语法 `class 名字(父类)`，跟"要传什么"无关。

**括号②比括号③多一个 `self`。** 你写 `GPT(35, 3)` 时 Python 做了两件事：先造一个空壳实例，再把这个空壳当**第一个参数**传进去。等价于：

```python
GPT.__init__(空壳, 35, 3)
```

所以**方法定义时永远要写 `self`，调用时永远不传**。

**括号①里那个 `nn.Module` 到底带来了什么？** 它是 PyTorch 所有层的基类，管着一套"记账机制"——见 §2.14。

### 1.8 `for` 循环 / `range` / `enumerate`

```python
for i in range(3):                 # range(3) = 0, 1, 2
    print(i)

for i, ch in enumerate(chars):     # enumerate 同时给"序号 + 元素"
    print(i, ch)                   # 0 a / 1 b / ...
```

`range(n)` 产生 0 到 n-1 的整数序列。`enumerate(列表)` 把列表变成 `(序号, 元素)` 的一对对，方便同时拿到下标和值。

### 1.9 切片 slicing

```python
a = [0, 1, 2, 3, 4]
a[:3]     # [0, 1, 2]   —— 前 3 个
a[3:]     # [3, 4]      —— 第 3 个及之后
a[1:3]    # [1, 2]      —— 下标 1 到 2（左闭右开）
```

`a[start:end]` 取一段，**左闭右开**（含 start，不含 end）。省略 start 表示从 0 开始，省略 end 表示到结尾。

---

## 2. PyTorch 部分

### 2.1 张量 Tensor —— 最基本的数据结构

```python
import torch
x = torch.tensor([1, 2, 3])         # 从列表创建
x.shape                             # torch.Size([3])
x.dtype                             # torch.int64
```

Tensor 是 PyTorch 的多维数组，是模型之间流转的唯一数据形式。它有两个关键属性：

- **shape（形状）**：几维、每维多大，如 `(B, T, C)` 表示"批量、长度、维度"。
- **dtype（数据类型）**：`torch.long` = 64 位整数（用于索引），`torch.float32` = 32 位浮点（默认，用于权重和计算）。

模型**只能吃 Tensor**，所以文本要先 `encode` 成整数、再 `torch.tensor(...)` 转成张量。

### 2.2 创建张量

```python
torch.tensor([1,2,3], dtype=torch.long)  # 从列表，指定类型
torch.arange(5)                          # [0,1,2,3,4]
torch.ones(2, 3)                         # 全 1，形状 2×3
torch.tril(torch.ones(4, 4))             # 下三角矩阵（右上角清零）
torch.randint(100, (2, 3))               # 形状 2×3 的随机整数（0~99）
torch.stack([a, b])                      # 把多个张量沿新的一维堆叠起来
torch.cat((a, b), dim=1)                 # 沿指定维度拼接
```

### 2.3 形状操作：`.view()` / `.transpose()`

```python
q = q.view(B, T, self.n_head, self.head_dim)  # 改形状
q = q.transpose(1, 2)                          # 交换第 1、2 维
y = y.contiguous().view(B, T, C)               # transpose 后先 contiguous 再 view
```

（这几个字母都是 `mini_gpt.py` 里的形状名：`B` = batch 批大小 32，`T` = 序列长度 ≤ 50，`C` = 通道数 96，`n_head` = 注意力头数 6，`head_dim` = 每个头的维度 16——`C = n_head × head_dim`。详见 `01-mini-gpt-reading.md` §1。）

- `.view(...)`：把张量**换个形状**，元素总数必须不变，只是重新排列维度。
- `.transpose(d0, d1)`：**交换两个维度**。注意它只是改了维度顺序，内存没重排，所以之后再 `.view()` 前常要 `.contiguous()` 让内存连续。
- 这些操作都不改数据本身，只改"怎么看它"。

```python
x = torch.arange(6)          # [0,1,2,3,4,5]，形状 (6,)
x.view(2, 3)                 # 形状变 (2,3)，数据顺序不变
# → tensor([[0, 1, 2],
#           [3, 4, 5]])

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])   # 形状 (2,3)
x.transpose(0, 1)               # 形状变 (3,2)
# → tensor([[1, 4],
#           [2, 5],
#           [3, 6]])
```

### 2.4 矩阵乘法 `@`

```python
att = q @ k.transpose(-2, -1)   # 矩阵乘
y = att @ v
```

`@` 是矩阵乘法。`-1` 指最后一维，`-2` 指倒数第二维，所以 `k.transpose(-2, -1)` 是"交换最后两维"（转置）。

```python
a = torch.tensor([[1, 2],
                  [3, 4]])   # 2×2
b = torch.tensor([[5, 6],
                  [7, 8]])   # 2×2
a @ b
# → tensor([[19, 22],
#           [43, 50]])     # 左上角 19 = 1*5 + 2*7，就是"行×列再求和"
```

### 2.5 广播 broadcast

```python
x = self.wte(idx) + self.wpe(pos)   # (B,T,C) + (T,C) → 自动变成 (B,T,C)；wte = word token embedding（词 token 嵌入），wpe = word position embedding（词位置嵌入）
```

两个形状不同的张量相加时，PyTorch 会自动把小的那个"扩展"成大的形状，这叫广播。这里 `wpe(pos)` 是 `(T,C)`，自动扩成 `(B,T,C)`，相当于"每一批都用同一份位置编码"。

```python
a = torch.ones(2, 3)             # 2×3，全是 1
b = torch.tensor([10, 20, 30])   # 长度 3 的一维
a + b
# → tensor([[11, 21, 31],
#           [11, 21, 31]])       # b 被自动"复制"成两行，再逐元素相加
```

**广播的两条规则**（`(B,T,C) + (T,C)` 就是靠这个对上的）：

1. **右对齐**比维度——从最后一维往前一维一维比，不是从左往右。`(T,C)` 补在右边，前导缺失的维度**隐式当成 1**。
2. 每一维：**相等** 或者 **其中一个是 1** → 通过；那个 1 会被**拉伸**成另一边的大小。

```
wte(idx)  (32, 50, 96)
wpe(pos)       (50, 96)      ← 缺了前导维
          ────────────────
右对齐       32   50  96     ← 缺的那维当 1，1 → 32（拉伸）

逐维： dim −1:  96 vs 96  相等 ✓
       dim −2:  50 vs 50  相等 ✓
       dim −3:  32 vs (1) 拉伸 ✓
```

（这里 32 / 50 / 96 就是 `mini_gpt.py` 里的 B / T / C。**三个数刻意取成两两不同**，就是为了让你看形状时不用猜哪一维是哪个——见 `01-mini-gpt-reading.md` §1。）

> ⚠️ **广播是静默的，这里有个坑。** 如果某一维是 1，它会**无声地**被拉伸——形状能对上、不报错，但语义可能已经错了。
>
> ```python
> a = torch.ones(2, 3, 2)
> b = torch.ones(1, 2)      # 倒数第二维是 1
> (a + b).shape             # → (2, 3, 2)，跑得通，但每个位置加的是同一份东西
> ```
>
> 所以看到形状不同的相加，**别只看"跑没跑通"，要问：它在哪一维上被拉伸了？那个拉伸是我想要的吗？** 完整例子（含三种写法对照）见 `01-mini-gpt-reading.md` §5.2。

### 2.6 常用层 `nn.*`

```python
nn.Embedding(vocab_size, n_embd)      # 嵌入表：输入整数索引 → 输出对应行向量
nn.Linear(n_embd, 3 * n_embd)         # 全连接（公式见下）
nn.LayerNorm(n_embd)                  # 层归一化
nn.Sequential(block1, block2, ...)    # 顺序容器：按顺序执行里面的层
```

- `nn.Linear(in, out)` 内部有**可训练参数 W 和 b**，`y = x @ W.T + b`。
- `nn.Embedding(num, dim)` 是一张 `num × dim` 的表，输入一个整数下标就吐出表里对应的一行。
- 这些层都继承 `nn.Module`，所以**它们内部的参数会被自动收集**，训练时能更新（这套记账机制见 §2.14）。

### 2.6.1 三个前缀：`nn.` / `self.` / `torch.`

这三个东西容易混，而且**混起来会直接读错代码**：

```python
self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # ① nn. 在这里出现（创建层）
...
q, k, v = self.c_attn(x).split(...)           # ② 用的时候只剩 self.，nn. 不见了
pos = torch.arange(T, device=idx.device)      # ③ torch.：调用一个函数
```

| 前缀 | 是什么 | 出现在哪 |
|---|---|---|
| `nn.` | PyTorch 自带的一个**层**（`nn.Linear` / `nn.Embedding` / `nn.LayerNorm` / `nn.Sequential`） | **只在 `__init__` 里创建时出现一次**，之后赋给 `self.xxx` |
| `self.` | "**这个实例自己**的属性" | 创建之后处处都用 |
| `torch.` | PyTorch 的一个**函数**（不是层）：`torch.arange` / `torch.tril` / `torch.randint` / `torch.cat` | 算东西的时候 |

**两个常见的误读，都得避开：**

**误读一："`forward` 里没写 `nn.`，所以那不是层。"** 错。`nn.Linear` 在 `__init__` 里创建完就改名叫 `self.c_attn` 了，`forward` 里当然看不到 `nn.`。

**误读二："`self.` 开头的就是参数。"** 错。`self.` 只说明"挂在谁身上"，跟可不可学无关：

```python
self.n_head     = n_head             # self.，但是个整数，无参数
self.head_dim   = n_embd // n_head   # self.，也是整数
self.block_size = block_size         # self.，还是整数

self.register_buffer('mask', torch.tril(...))   # self.，是张量，但登记为常量，不训练
```

**"有 `nn.` 就可学"同样不成立**——`nn.ReLU`、`nn.Dropout` 也是 `nn.` 开头，却没有参数。

**那到底怎么判断？唯一权威的判据：**

```python
model.parameters()     # 出现在里面的，才是会被优化器更新的
```

想自己验某个属性可不可学，就这么测：

```python
print(any(p is model.blocks[0].attn.mask for p in model.parameters()))   # False → 那个 mask 不可学
```

### 2.7 `register_buffer`

```python
self.register_buffer('mask', torch.tril(...))
```

`mask` 是模型要用的固定矩阵（因果掩码），但它**不是可训练参数**、也不该被优化器更新。`register_buffer` 就是声明"这是随模型一起保存/迁移的常量，但不是参数"。

> 它和普通赋值 `self.mask = ...` 的区别，正是 §2.6.1 里那个 `self.mask` 反例：**buffer 会进 `state_dict`，但不会进 `parameters()`。** 而直接赋值的普通张量，两样都不进。
>
> （`state_dict()` = 模型所有参数和 buffer 的"名字 → 张量"字典，存盘/加载用的就是它；完整说明见 §2.14.3。）

### 2.8 激活函数与损失函数

```python
F.gelu(x)                             # GELU（Gaussian Error Linear Unit）激活
F.softmax(logits, dim=-1)             # softmax，沿最后一维归一化成概率（logits = 未归一化的原始打分）
F.cross_entropy(logits, targets)      # 交叉熵损失（内部已含 log_softmax）
```

`F.xxx` 里都是纯函数（没有可训练参数）。注意 `F.cross_entropy` 内部已经做了 log_softmax，所以**喂给它的是原始 logits，不是概率**。

```python
p = F.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
# → tensor([0.0900, 0.2447, 0.6652])
# 三个数加起来正好 = 1，这就是"概率"的含义；越大的数分到的概率越高
```

> **最容易踩的坑：`dim` 指的不是"结果里和为 1 的那条轴"，而是"被折叠掉的那条轴"**——看 `x.shape` 少掉第几个数字就知道。

```python
x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])      # shape (2,3)：dim0=2 行，dim1=3 列

x.sum(dim=0)      # → [5., 7., 9.]，shape 变 (3,)：第 0 个数字"2"没了 → 折叠掉的是【行】
x.sum(dim=1)      # → [6., 15.]，shape 变 (2,)：第 1 个数字"3"没了 → 折叠掉的是【列】

F.softmax(x, dim=0)   # 沿行折叠 → 同一列上下两个数竞争 → 每【列】的和 = 1
# → [[0.0474, 0.0474, 0.0474],
#     [0.9526, 0.9526, 0.9526]]

F.softmax(x, dim=1)   # 沿列折叠 → 一行内三个数竞争 → 每【行】的和 = 1
# → [[0.0900, 0.2447, 0.6652],
#     [0.0900, 0.2447, 0.6652]]
```

一句话记牢：**dim 指向谁，谁就被"吃掉"；和等于 1 的，是它对面那条轴。** `dim=-1` 就是最后一维（见 2.4）。Transformer 的注意力分数矩阵最后两维是 `(query 数, key 数)`，要让一个 query 把概率分给所有 key，就沿最后维归一化——所以你会在代码里反复看到 `softmax(..., dim=-1)`。

### 2.9 自动求导：`.backward()` 与 `@torch.no_grad()`

```python
loss.backward()          # 反向传播：算出每个参数的梯度，存到 param.grad 里

@torch.no_grad()         # 装饰器：写在 def 上一行，等价于 generate = torch.no_grad()(generate)
def generate(self, ...):
    ...
```

`.backward()` 触发自动求导，算出的梯度存在每个参数的 `.grad` 上。**推理/生成阶段不需要梯度**，用 `@torch.no_grad()` 关掉，省内存也更快。

（注意区分：写在 `def` 上一行的 `@名字` 是**装饰器**，和 §2.4 那个矩阵乘法的 `@` 是两个完全不同的符号。）

### 2.10 优化器：`zero_grad` / `step`

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # 建优化器，管所有参数
optimizer.zero_grad(set_to_none=True)   # 把上一轮的梯度清零
loss.backward()                          # 算这一轮的梯度
optimizer.step()                         # 用梯度更新参数
```

一次训练迭代的固定三步：**清零 → 反传 → 更新**。`model.parameters()` 返回模型里所有可训练参数。

### 2.11 采样 `torch.multinomial`

```python
next_token = torch.multinomial(probs, num_samples=1)  # 按概率采样 1 个
```

`probs` 是一串概率（和为 1），`torch.multinomial` 按这些概率随机抽一个下标。概率高的更常被抽到，但低的也有机会——这就是生成文本的"随机性"来源。

```python
probs = torch.tensor([0.7, 0.2, 0.1])
torch.multinomial(probs, num_samples=1)   # 大概率抽到 0，但 1、2 也有机会
```

### 2.12 把张量变回 Python 值

```python
loss.item()        # 单元素张量 → Python 浮点数
idx.tolist()       # 张量 → Python 列表
p.numel()          # 张量元素总数（数参数量时用）
```

`.item()` 只能用于**只有一个元素**的张量（比如标量 loss）；`.tolist()` 把整个张量转成嵌套列表。

### 2.13 随机性与 `torch.manual_seed`

```python
torch.manual_seed(1337)   # mini_gpt.py:20 —— 开头设置区的第一行（在 import 之后，建模型之前）
```

给**随机数发生器定一个起点**。计算机里的随机是**伪随机**：从一个种子出发，用固定公式往前推出一串数。同一个种子 → 同一串数；不设种子 → PyTorch 拿系统时间之类的当种子，每次都不一样。

**mini_gpt.py 里有三处随机，全都归它管：**

| 位置 | 随机用在干嘛 |
|---|---|
| `nn.Embedding` / `nn.Linear` 的内部 | 初始化权重（**文件里看不见**，但这是最大的一处） |
| `mini_gpt.py:195` `torch.randint` | 每个 step 抽哪 32 条数据当一批 |
| `mini_gpt.py:185` `torch.multinomial` | 生成时按概率掷骰子选下一个字 |

**实测一：种子让结果可复现。** 建一台 `nn.Embedding(35, 96)`，看权重的前 3 个数：

```
不设种子，连建两台：  [-0.1311,  0.4792, -0.5840]    [-0.7187, ...]              ← 两台不同
每次先设 1337：       [ 0.1808, -0.0700, -0.3596]    [ 0.1808, -0.0700, -0.3596] ← 完全相同
```

**实测二：种子定的是"起点"，不是"结果"。** 同一次播种之后**连着**建两台，第二台就不同了——随机数是一条往前流的河，`manual_seed` 只是把河口定死：

```
torch.manual_seed(1337)
  第 1 台  [0.1808, -0.0700, -0.3596]
  第 2 台  [0.1397,  0.2515, -0.8921]   ← 往前走了
```

**实测三：整条训练路径也是可复现的**（同一份数据、同一个模型结构，跑 3 步）：

```
不设种子，跑三遍：   1.816174   2.273806   1.749255   ← 三个数都不一样
都设 1337，跑三遍：  2.087231   2.087231   2.087231   ← 完全一致
```

**它买到的是"可归因"。** 改了 A，loss 变了，你能确定是 A 改的；没有种子，你分不清是改动的功劳还是抽签的运气。这就是一个"最小骨架"也要开头就放它的原因。

**`1337` 是什么？** 黑客圈的老梗：leetspeak 里用数字代字母，`1337` 读作 **leet**（elite 的缩写），Karpathy 惯用。**数字本身没有含义**，写 42、0、1234 都一样，只要别改来改去。

**两个边界：**

- 它管的是 **CPU 的默认随机源**；GPU 各有自己的发生器，要另外调 `torch.cuda.manual_seed_all(n)`。
- 它保证"同一次运行、同一台机器、同一个 PyTorch 版本"可复现，**跨版本、跨硬件不保证**——硬件不同，某些算子的实现不同，随机数被消耗的顺序会变。

### 2.14 `nn.Module` 的记账机制

**`nn.Module` 是什么？** PyTorch 里所有神经网络模块的**基类**。（`nn` 是 neural network 的缩写，`torch.nn` 里全是层。）`nn.Linear`、`nn.Embedding`、`nn.LayerNorm`，以及你自己写的 `GPT` / `Block` / `MLP`，全是它的子类。

它在 §1.7 里出现过一次（`class GPT(nn.Module)` 的括号①），但当时只说了"那是父类"。**这一节讲那个父类具体带来了什么。**

#### 2.14.1 三个抽屉：登记到底记了什么

它自己不计算任何东西，**提供的是一套记账机制**。把 `mini_gpt.py` 建好的模型打开，看它的"子模块表"（实测）：

```
wte       Embedding
wpe       Embedding
blocks    Sequential      ← 里面还套着 3 个 Block
ln_f      LayerNorm
lm_head   Linear
```

（这只是最外面一层。整棵树——包括 `blocks` 里套着的 `Block`、`Block` 里套着的 `attn` / `mlp`——见 §2.14.2。）

你从没写过"把这些层登记到一个列表里"，但 `nn.Module` 全记住了。机制是**它改写了 `self.xxx = ...` 这个赋值动作**，赋值时先看右边是什么：

| 赋的值是 | 记到哪 | 例子 |
|---|---|---|
| `nn.Module` 的子类（也就是"层"） | **子模块表** | `self.ln1 = nn.LayerNorm(n_embd)` |
| `nn.Parameter`（声明为可训练的张量） | **参数表** | 本文件没直接用，`nn.Linear` 内部就是这么建的 |
| 其它（`int` / `str` / 普通张量） | 不记账，就是个普通属性 | `self.n_head = 6`、`self.block_size = 50` |

**"登记"具体是个什么动作？就是往一本字典里写一条 `名字 → 对象`。** 这几个抽屉都是最普通的 `dict`（实测 `m._modules`、`m._parameters`、`m._buffers` 的 `type` 都是 `dict`）：

```
_modules      ['wte', 'wpe', 'blocks', 'ln_f', 'lm_head']   ← 根节点自己直接挂的 5 个
_parameters   []
_buffers      []              ← 3 个 mask 不在根上，藏在各 Block 的 attn 里（见下方那棵树）
```

名字是**分层**的，每一层只管自己那一层：`m.blocks._modules` 的键是 `['0','1','2']`，`m.blocks[0].attn._modules` 是 `['c_attn','c_proj']`——`state_dict()` 里的 `blocks.0.attn.c_attn.weight`，就是把这些层名用点连起来。

登记还是个**动词**，而且即时生效：对建好的模型手动执行一次 `m.register_parameter('w_new', nn.Parameter(torch.randn(5,5)))`，`m.parameters()` 立刻从 42 个变成 43 个（实测）。这跟芯片里的"进网表"是一回事：**版图上画了个器件，但没进 netlist，仿真和 STA 就都看不见它。**

#### 2.14.2 登记发生在哪一刻：把构造过程走一遍

**结论先说：不在 `super().__init__()`，而在之后每一次赋值。** 一个小类实测，每步都把三本账打出来：

```python
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()                            # ①
        self.n = 3                                    # ②
        self.lin = nn.Linear(2, 2)                    # ③
        self.w = nn.Parameter(torch.randn(2, 2))      # ④
        self.t = torch.randn(2, 2)                    # ⑤
        self.register_buffer('buf', torch.zeros(2))   # ⑥
```

执行到每一步时，三本账里各有什么（实测）：

| 执行到 | `_modules` | `_parameters` | `_buffers` |
|---|---|---|---|
| ① `super().__init__()` 刚返回 | `[]` | `[]` | `[]` |
| ② `self.n = 3`（int） | `[]` | `[]` | `[]` |
| ③ `self.lin = nn.Linear(2,2)` | `['lin']` | `[]` | `[]` |
| ④ `self.w = nn.Parameter(...)` | `['lin']` | `['w']` | `[]` |
| ⑤ `self.t = 普通张量` | `['lin']` | `['w']` | `[]` |
| ⑥ `register_buffer('buf', ...)` | `['lin']` | `['w']` | `['buf']` |

**① 那一行跑完，三本账全是空的**——`super().__init__()` 只干一件事：把账本摆出来。登记是从 ③ 开始，一行一行攒出来的。

**换成真模型，这棵"账本树"长这样。** 把 `mini_gpt.py` 建好的模型整个倒出来（实测；第 1、2 层与第 0 层完全一样，略）：

```
GPT
├─ _modules['wte'] = Embedding
│  └─ _parameters: weight
├─ _modules['wpe'] = Embedding
│  └─ _parameters: weight
├─ _modules['blocks'] = Sequential
│  ├─ _modules['0'] = Block
│  │  ├─ _modules['ln1'] = LayerNorm
│  │  │  └─ _parameters: weight, bias
│  │  ├─ _modules['attn'] = CausalSelfAttention
│  │  │  ├─ _modules['c_attn'] = Linear
│  │  │  │  └─ _parameters: weight, bias
│  │  │  ├─ _modules['c_proj'] = Linear
│  │  │  │  └─ _parameters: weight, bias
│  │  │  └─ _buffers: mask               ← 唯一"不训练"的那一栏
│  │  ├─ _modules['ln2'] = LayerNorm
│  │  │  └─ _parameters: weight, bias
│  │  └─ _modules['mlp'] = MLP
│  │     ├─ _modules['c_fc'] = Linear
│  │     │  └─ _parameters: weight, bias
│  │     └─ _modules['c_proj'] = Linear
│  │        └─ _parameters: weight, bias
│  ├─ _modules['1'] = Block               ← 与第 0 层完全一样，略
│  └─ _modules['2'] = Block               ← 同上，略
├─ _modules['ln_f'] = LayerNorm
│  └─ _parameters: weight, bias
└─ _modules['lm_head'] = Linear
   └─ _parameters: weight, bias
```

三个值得盯一眼的地方：

- **每往下一层缩进，就多一级"名字的所有权"。** `GPT` 里根本没有 `attn` 这个名字——是 `GPT.blocks` 登记了 `Sequential`，`Sequential.0` 登记了 `Block`，`Block.attn` 才登记 `CausalSelfAttention`。**名字逐级下放，拼起来才唯一。**
- **层是在子层之后登记的。** 看 `Block.attn` 这一支，实际写入的先后是：

  ```
      c_attn → _modules['c_attn'] = Linear
      c_proj → _modules['c_proj'] = Linear
    attn   → _modules['attn']  = CausalSelfAttention   ← 里面的都记完了，它才被记进 Block
  ```

  像盖楼，**先盖完，再挂户口**。
- **数字对得上：三本账一共 77 条 = 32 个层 + 42 个参数张量 + 3 个 buffer**；此外还有 68 次普通赋值（`int` / `str` / `bool` / `None`…），一条都没记。**每个参数都来自一次赋值动作（包括 `nn.Linear` 内部替你做的那次），没有魔法。**

> **一个容易看漏的细节：不是所有登记都走 `__setattr__`。** 盯住 `Sequential` 那三个子层——如果只挂 `__setattr__` 的钩子去数，会数出 29 个层，比真实的 32 少 3 个。少掉的正是它们：`nn.Sequential` 用的是 `add_module()`，它**直接写 `self._modules[name]`**，不经过赋值动作。（`register_buffer` 也一样，直接写 `_buffers`。）**入口不止一个，但账本是同一本。**

#### 2.14.3 这套账换来什么：从遍历到"为什么必须是基类"

**这套账直接兑换成最后那几行代码：**

| 写法 | 靠这套账做什么 | 实测 |
|---|---|---|
| `model.parameters()` | 递归走完整棵树，一次拿到所有可训练参数 | 347,267 个，一行交给优化器 |
| `model.to(device)` | 同样递归，一次把所有张量搬到同一台设备 | `modules()` 展开有 33 个（32 个子模块 + 根节点自己） |
| `model.state_dict()` | 存盘/加载用的账本，键就是这棵树的路径 | `wte.weight`、`blocks.0.attn.c_attn.weight`、`ln_f.bias`… |
| `model.train()` / `model.eval()` | 递归设一个"训练 / 评测"标志位 | `estimate_loss` 靠它切到评测模式 |
| `model(x)` | `nn.Module` 实现了 `__call__`，转手去调 `forward` | `model(x)[0]` 与 `model.forward(x)[0]` 逐元素相等 |

**那"哪些要学、哪些不要学"，是父类在判断吗？——不是。**

父类只做**分拣**：看等号右边是什么类型，抄进对应的本子。它没有"这个该不该学"的逻辑。**决定权在你手上**——你写 `nn.Parameter` 就进参数本，写 `register_buffer` 就进 buffer 本，写普通张量就哪儿都不进。

真正做判断的是**查询的一方**：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)   # 给什么就更新什么
```

实测（一台 `blocks = 3 × (Linear + 一个 mask buffer)` 的小模型）：

```
parameters() 名字: ['blocks.0.c.weight', 'blocks.0.c.bias', 'blocks.1.c.weight', ...]   ← 6 个
buffers()    名字: ['blocks.0.mask', 'blocks.1.mask', 'blocks.2.mask']                  ← 3 个
优化器拿到的张量数: 6                ← buffer 一个都没有
'mask' 在 parameters() 里吗: False
```

所以这两栏的差别不只在"学不学"：`_buffers` 不学，但**要跟着搬设备、跟着存盘**（`state_dict()` 里两者都在）。

**`super().__init__()` 必须是每个 `__init__` 的第一行**。漏掉它，后面的 `self.wte = ...` 无处登记，报的正是：

```
AttributeError: cannot assign module before Module.__init__() call
```

——"`Module.__init__()` 之前不许赋值模块"，它管的恰好就是赋值这个动作。

> **为什么 `nn.Module` 必须做"基类"，而不是做成一个函数？**
>
> 一句话：**`self.wte = ...` 不是"往内存里写"，而是一次方法调用。** Python 里 `对象.属性 = 值` 等价于 `type(对象).__setattr__(对象, '属性', 值)`——每次赋值，Python 都会去执行一个叫 `__setattr__` 的方法。
>
> `nn.Module` 要插手的正是这个动作：赋值时看一眼右边是不是"层"或"参数"，是就顺手记一笔。**而 Python 找 `__setattr__` 只在"类型"上找**（沿 `type(对象)` 的继承链往上走），不看实例——所以想插手，唯一的办法就是继承 `nn.Module`，把自己挂进那条链子。做成函数就完全插不上手：`self.wte = ...` 那一行根本不认识它。
>
> 那为什么非要"自动"？因为漏登记的后果是**静默**的：一个参数照样参与前向、照样有梯度，但 `parameters()` 里没有它，优化器永远不更新它，**而且不报错**。所以设计目标不是"少写几行"，而是"让你不可能忘记"——而你最自然的写法 `self.x = ...`，恰好就是登记本身。
>
> 顺带，`model(x)` 能跑也是同一回事：`__call__` 也是类型上的钩子。Python 的语法糖（`obj()`、`obj[k]`、`a + b`）背后全是类型上的特殊方法——**所以任何想挂钩语法的框架，都必须是类。**

> 这套账在前面已经用过：§2.6（层内部参数自动收集）、§2.6.1（`model.parameters()` 是判断"可不可学"的唯一权威）、§2.7（`register_buffer` 走的是同一套账，只是登记进"不训练"的那一栏）。

---

## 3. 一眼看懂"训练一趟"的完整流程

```python
for step in range(max_iters):        # 循环 max_iters 次
    xb, yb = get_batch('train')      # 取一批数据 (x=上文, y=下一位)
    logits, loss = model(xb, yb)     # 前向：算出预测和损失
    optimizer.zero_grad(set_to_none=True)  # 1. 清梯度
    loss.backward()                         # 2. 反传，算梯度
    optimizer.step()                        # 3. 用梯度更新权重
```

---

## 4. 动手试试：一个可运行的"小操场"

把下面这段存成 `playground.py`，然后 `python3 playground.py`；或者打开 `python3` 逐行粘贴，就能亲眼看到每个操作的结果：

```python
import torch
import torch.nn.functional as F

print("1. view：改形状，数据顺序不变")
x = torch.arange(6)
print(x.view(2, 3), "\n")

print("2. transpose：交换两维")
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.transpose(0, 1), "\n")

print("3. @：矩阵乘法")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(a @ b, "\n")

print("4. 广播：形状不同也能加")
print(torch.ones(2, 3) + torch.tensor([10, 20, 30]), "\n")

print("5. softmax：输出是概率，和为 1")
p = F.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0)
print(p, "和 =", p.sum(), "\n")

print("6. Embedding：输入下标，查出一行向量")
emb = torch.nn.Embedding(4, 3)          # 4 个词，每个词 3 维
print("查第 0、2 号词：\n", emb(torch.tensor([0, 2])))
```

跑起来最直观。跑不通、或者哪行输出和你预期不一样，把那行贴给我。
