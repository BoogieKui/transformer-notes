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