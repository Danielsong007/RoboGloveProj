import numpy as np
import matplotlib.pyplot as plt

def nonlinear_control(pres):
    p_left = 0
    y_left = 0
    p_right = 1000
    y_right = 1.2
    
    # 中间直线段范围
    mid_start = 600
    mid_end = 900
    y_mid = 1.0
    
    if pres <= mid_start:
        # 左侧三次曲线区域 [0, 600]
        # 使用三次曲线平滑连接 (0,0) 到 (600,1.0)
        # 在连接点处导数为0确保平滑
        t = (pres - p_left) / (mid_start - p_left)  # t从0到1
        coef = y_left + (y_mid - y_left) * (3*t**2 - 2*t**3)
        
    elif pres <= mid_end:
        # 中间直线段 [600, 900]
        coef = y_mid
        
    else:
        # 右侧三次曲线区域 [900, 1000]
        # 使用三次曲线平滑连接 (900,1.0) 到 (1000,1.2)
        # 在连接点处导数为0确保平滑
        t = (pres - mid_end) / (p_right - mid_end)  # t从0到1
        coef = y_mid + (y_right - y_mid) * (3*t**2 - 2*t**3)
    
    return coef

# 测试关键点
test_points = [0, 200, 400, 600, 700, 800, 900, 950, 1000]
print("关键点测试结果:")
for p in test_points:
    print(f"p={p}: {nonlinear_control(p):.3f}")

# 绘制曲线
pres_values = np.linspace(0, 1000, 1000)
output_values = [nonlinear_control(p) for p in pres_values]

plt.figure(figsize=(12, 6))
plt.plot(pres_values, output_values, 'b-', linewidth=3, label='三次曲线连接')

# 标记关键点和区域
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='p_left=0')
plt.axvline(x=600, color='g', linestyle='--', alpha=0.7, label='mid_start=600')
plt.axvline(x=900, color='g', linestyle='--', alpha=0.7, label='mid_end=900')
plt.axvline(x=1000, color='r', linestyle='--', alpha=0.5, label='p_right=1000')

plt.axhline(y=0, color='orange', linestyle=':', alpha=0.5, label='y_left=0')
plt.axhline(y=1.0, color='purple', linestyle='-', alpha=0.5, label='y_mid=1.0')
plt.axhline(y=1.2, color='orange', linestyle=':', alpha=0.5, label='y_right=1.2')

# 标记关键点
key_points = [0, 600, 900, 1000]
key_values = [0, 1.0, 1.0, 1.2]
plt.scatter(key_points, key_values, color='red', s=100, zorder=5)

plt.xlabel('Pressure', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.title('三次曲线平滑连接的水平直线段控制函数', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()