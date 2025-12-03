import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('sensor_data.csv')

# 选择前三列数据（跳过第一行标题）
col1 = df.iloc[1:, 0]  # 第一列数据
col2 = df.iloc[1:, 1]  # 第二列数据
col3 = df.iloc[1:, 2]  # 第三列数据
col4 = df.iloc[1:, 3]  # 第三列数据
col5 = df.iloc[1:, 4]  # 第三列数据
col6 = df.iloc[1:, 5]  # 第三列数据
col7 = df.iloc[1:, 6]  # 第三列数据

# 创建横轴（数据行数）
x = range(len(col1))

# 绘制图形
plt.figure(figsize=(12, 6))

# 在同一张图中绘制三列数据，使用不同颜色
plt.plot(x, col1, 'b-', label='第一列数据', linewidth=1.5)
plt.plot(x, col2*1000, 'r-', label='第二列数据', linewidth=1.5)
plt.plot(x, col3, 'g-', label='第三列数据', linewidth=1.5)
# plt.plot(x, (col5-1951029190)/10000, 'k-', label='第三列数据', linewidth=1.5)


plt.xlabel('数据行数')
plt.ylabel('数值')
plt.title('前三列数据对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()