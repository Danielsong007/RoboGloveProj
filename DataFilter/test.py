import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据
df = pd.read_csv('DataFilter/sensor_data_test.csv')
df=df[6000:9000]
df_mode2 = df[df['mode'] == 2].copy()

# 2. 提取数据
F = df_mode2['Rope_S'].values.astype(float)
P = df_mode2['Touch_S'].values.astype(float)
A = df_mode2['Acceleration_Smooth'].values.astype(float)
g = 9.81

# 3. 固定参数
k_acc_fixed = 1.5
k_f_fixed = 0.1

# 4. 模型函数
def model_residuals(params, F_data, P_data, A_data):
    """残差: 0.1*F + k_p*P - m*g - m*1*A"""
    m, k_p = params
    return k_f_fixed*F_data + k_p*P_data - m*g - m*k_acc_fixed*A_data

# 5. 优化
initial_guess = [10.0, 0.1]
bounds = ([0.01, 0.001], [50.0, 100.0])

result = least_squares(model_residuals, initial_guess, 
                      args=(F, P, A),
                      bounds=bounds)

# 6. 结果
m_opt, k_p_opt = result.x
residuals = result.fun
print(np.mean(residuals),residuals[0:20])
F_total_measure = k_f_fixed*F + k_p_opt*P
F_total_model = m_opt*g + m_opt*k_acc_fixed*A
r2 = 1 - np.sum(residuals**2) / np.sum((F_total_measure - np.mean(F_total_measure))**2)

# 7. 输出
print(f"固定参数: k_acc={k_acc_fixed}, k_f={k_f_fixed}")
print(f"估计参数: m={m_opt:.4f}kg, k_p={k_p_opt:.5f}")
print(f"模型: {m_opt:.4f}*A = {k_f_fixed:.3f}*F + {k_p_opt:.4f}*P - {m_opt*9.81:.4f}")
print(f"精度: R²={r2:.4f}, 残差RMS={np.sqrt(np.mean(residuals**2)):.4f}")

# 8. 保存
pd.DataFrame([{
    'mass_kg': m_opt,
    'k_acc': k_acc_fixed,
    'k_F': k_f_fixed,
    'k_P': k_p_opt,
    'R2': r2
}]).to_csv('mass_estimation_fixed.csv', index=False)