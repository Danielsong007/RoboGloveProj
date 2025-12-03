import numpy as np
import matplotlib.pyplot as plt



# def nonlinear_control(pres):
#     p_left=200
#     p_right=1600
#     p_cent=(p_left+p_right)/2
#     if pres < p_cent:
#         pres=(pres - p_left) / 150
#         coef = 1 / (1+np.exp(-pres))
#     else:
#         pres=(pres - p_right) / 150
#         coef = 1 + 1*1 /(1+np.exp(-pres))
#     return coef

# def nonlinear_control(pres):
#     p_left = 200
#     y_left = 0.2
#     p_right = 1600
#     y_right = 2
#     coef = y_left + (y_right-y_left)/(p_right-p_left)*(pres-p_left)
#     coef = np.clip(coef, 0,2)
#     return coef*1500

# def nonlinear_control(pres):
#     p_left = 200
#     y_left = 0.8
#     p_right = 1000
#     y_right = 1.2
#     p_cent = (p_left + p_right) / 2
#     y_cent = 1.0
#     if pres <= p_cent:
#         x_norm = (pres - p_left) / (p_cent - p_left)
#         a = y_left - y_cent
#         b = 2 * (y_cent - y_left)
#         coef = a * x_norm**2 + b * x_norm + y_left
#     else:
#         x_norm = (pres - p_cent) / (p_right - p_cent)
#         a = y_right - y_cent
#         b = 0
#         coef = a * x_norm**2 + b * x_norm + y_cent
#     return coef

def nonlinear_control(pres):
    p_left = 300
    y_left = 0
    p_right = 1500
    y_right = 1.5
    p_cent = (p_left + p_right) / 2  # 600
    mid_start = 500  # 中间直线段开始
    mid_end = 900    # 中间直线段结束
    if pres <= mid_start:
        # 左侧二次曲线区域
        x_norm = (pres - p_left) / (mid_start - p_left)
        a = y_left - 1.0  # 目标值改为1.0（中间直线段的值）
        b = 2 * (1.0 - y_left)
        coef = a * x_norm**2 + b * x_norm + y_left
    elif pres <= mid_end:
        coef = 1.0
    else:
        x_norm = (pres - mid_end) / (p_right - mid_end)
        a = y_right - 1.0  # 目标值改为1.0（中间直线段的值）
        b = 0
        coef = a * x_norm**2 + b * x_norm + 1.0
    coef = np.clip(coef, 0,2)
    return coef

# 创建压力值范围
pres_values = np.linspace(0, 2000, 1000)
Vgoal_N_values = [nonlinear_control(pres) for pres in pres_values]

# 绘制关系图
plt.figure(figsize=(12, 8))
plt.plot(pres_values, Vgoal_N_values, linewidth=2, color='blue')
plt.xlabel('Pressure (pres)', fontsize=12)
plt.ylabel('Vgoal_N', fontsize=12)
plt.title('Relationship between Vgoal_N and pres', fontsize=14)
plt.grid(True, alpha=0.3)

# 添加参考线
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=200, color='r', linestyle='--', alpha=0.7, label='Center pressure (300)')
plt.axvline(x=600, color='g', linestyle='--', alpha=0.7, label='Transition point (800)')
plt.axvline(x=1000, color='b', linestyle='--', alpha=0.7, label='Transition point (1200)')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()
