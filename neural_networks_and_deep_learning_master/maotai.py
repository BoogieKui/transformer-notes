import matplotlib.pyplot as plt  
import matplotlib.ticker as ticker  
  
# === 数据准备 (2010-2023) ===  
years = [str(y) for y in range(2010, 2024)]  
# 年末收盘价 (大概数据)  
stock_prices = [183, 193, 208, 128, 189, 218, 334, 697, 590, 1183, 1998, 2050, 1727, 1726]  
# 年度分红总额 (亿元)  
dividends = [21.69, 41.50, 66.64, 45.41, 54.95, 77.51, 85.26, 138.17, 182.64, 213.87, 242.36, 272.28, 325.49, 387.86]  
  
# === 设置画板 ===  
plt.figure(figsize=(12, 6), dpi=150)  
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei'] # 适配中文显示  
plt.rcParams['axes.unicode_minus'] = False  
  
# === 绘制双轴 ===  
ax1 = plt.gca() # 获取当前轴  
ax2 = ax1.twinx() # 创建共享X轴的第二个Y轴  
  
# 1. 绘制分红 (柱状图 - 代表公司业绩底座)  
bars = ax2.bar(years, dividends, color='#85C1E9', alpha=0.6, label='现金分红总额(亿元)', width=0.5)  
# 2. 绘制股价 (折线图 - 代表市场情绪)  
line, = ax1.plot(years, stock_prices, color='#C0392B', marker='o', linewidth=2.5, label='年末股价(元)')  
  
# === 细节标注 ===  
# 标注2013年黄金坑  
plt.annotate('2013年黄金坑\n(股价腰斩，分红依旧)',   
             xy=(3, 128), xytext=(4, 600),  
             arrowprops=dict(facecolor='black', shrink=0.05),  
             fontsize=10, color='black')  
  
# === 设置轴标签和标题 ===  
ax1.set_ylabel('股价 (元)', color='#C0392B', fontsize=12, fontweight='bold')  
ax2.set_ylabel('分红总额 (亿元)', color='#2980B9', fontsize=12, fontweight='bold')  
ax1.set_title('贵州茅台(2010-2023): 股价波动 vs 分红增长', fontsize=14, pad=20)  
ax1.grid(True, linestyle='--', alpha=0.3)  
  
# === 整合图例 ===  
lines, labels = ax1.get_legend_handles_labels()  
lines2, labels2 = ax2.get_legend_handles_labels()  
ax1.legend(lines + lines2, labels + labels2, loc='upper left')  
  
plt.tight_layout()  
plt.show()  
