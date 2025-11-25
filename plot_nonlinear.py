import numpy as np
import matplotlib.pyplot as plt


def nonlinear_control(pres):
    if pres > 300:
        pres=(pres - 400) / 300 * 10
        Vgoal_N = 1 / (1+np.exp(-pres)) * 11000 - 384
    else:
        pres=(pres - 200) / 300 * 10
        Vgoal_N = 1 / (1+np.exp(-pres)) * 5000 - 5000 + 178
    return Vgoal_N


# def nonlinear_control(pres):
#     if pres > 300:
#         pres=(pres - 400) / 300 * 10
#         Vgoal_N = 1 / (1+np.exp(-pres)) * 8000 - 279
#     else:
#         pres=(pres - 200) / 300 * 10
#         Vgoal_N = 1 / (1+np.exp(-pres)) * 5000 - 5000 + 178
#     return Vgoal_N

# 创建压力值范围
pres_values = np.linspace(0, 1500, 1000)
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
plt.axvline(x=300, color='r', linestyle='--', alpha=0.7, label='Center pressure (300)')
plt.axvline(x=800, color='g', linestyle='--', alpha=0.7, label='Transition point (800)')
plt.axvline(x=1200, color='b', linestyle='--', alpha=0.7, label='Transition point (1200)')
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()

# 打印一些关键点的值
key_points = [0, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1500]
print("关键点的Vgoal_N值:")
print("pres\tVgoal_N")
for p in key_points:
    v = nonlinear_control(p)
    print(f"{p}\t{v:.2f}")