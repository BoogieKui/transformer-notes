import numpy as np
import matplotlib.pyplot as plt
import mnist_loader_py3  # 假设 mnist_loader_py3.py 已正确实现

# 加载数据（使用 load_data() 或 load_data_wrapper()）
tr_d, va_d, te_d = mnist_loader_py3.load_data()  # 原始数据格式
# training_data, _, _ = mnist_loader.load_data_wrapper()  # 神经网络格式（可选）

# 选择一个样本（例如第0个样本）
sample_index = 0
image = tr_d[0][sample_index]  # 原始图像数据（形状为 (784,)）
label = tr_d[1][sample_index]  # 对应的标签

# 将展平的 784 维向量转换为 28x28 的图像
image_reshaped = image.reshape((28, 28))

# 可视化
plt.figure(figsize=(5, 5))
plt.imshow(image_reshaped, cmap='gray')  # 灰度显示
plt.title(f"Label: {label}")
plt.axis('off')  # 隐藏坐标轴
plt.show()