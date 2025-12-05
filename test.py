import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 读取CSV文件
df = pd.read_csv('sensor_data.csv')

# 获取第五列（索引4）的编码器原始数据
pos_raw = df.iloc[1:, 4].astype(float).values
t = np.arange(len(pos_raw))

# 设置采样频率
fs = 1000  # 请根据实际情况修改
dt = 1/fs

print(f"Data Information:")
print(f"  - Data points: {len(pos_raw)}")
print(f"  - Sampling frequency: {fs} Hz")
print(f"  - Sampling interval: {dt:.6f} s")
print(f"  - Data range: [{pos_raw.min():.3f}, {pos_raw.max():.3f}]")

# ============================================
# 1. 直接差分法（原始方法）
# ============================================
def direct_difference(pos, dt):
    """
    直接差分法求速度和加速度
    """
    n = len(pos)
    
    # 速度：后向差分
    vel_raw = np.zeros(n)
    vel_raw[1:] = (pos[1:] - pos[:-1]) / dt
    vel_raw[0] = vel_raw[1]  # 第一个点用第二个点的值
    
    # 加速度：速度的后向差分
    acc_raw = np.zeros(n)
    acc_raw[1:] = (vel_raw[1:] - vel_raw[:-1]) / dt
    acc_raw[0] = acc_raw[1]  # 第一个点用第二个点的值
    
    return vel_raw, acc_raw

# 计算直接差分的结果
vel_direct, acc_direct = direct_difference(pos_raw, dt)

# ============================================
# 2. 各种平滑方法
# ============================================
def smooth_differentiation(pos, dt=0.001, fc_pos=20, fc_vel=10, method='butterworth'):
    n = len(pos)
    fs = 1/dt
    
    if method == 'butterworth':
        b, a = signal.butter(2, fc_pos/(fs/2), 'low')
        pos_smooth = signal.filtfilt(b, a, pos)
        
        vel = np.gradient(pos_smooth, dt)
        b, a = signal.butter(2, fc_vel/(fs/2), 'low')
        vel_smooth = signal.filtfilt(b, a, vel)
        
        acc = np.gradient(vel_smooth, dt)
        acc_smooth = signal.filtfilt(b, a, acc)
        
    elif method == 'moving_avg':
        window_size = 7
        kernel = np.ones(window_size) / window_size
        
        pos_smooth = np.convolve(pos, kernel, mode='same')
        vel = np.gradient(pos_smooth, dt)
        vel_smooth = np.convolve(vel, kernel, mode='same')
        acc = np.gradient(vel_smooth, dt)
        acc_smooth = np.convolve(acc, kernel, mode='same')
        
    elif method == 'savgol':
        window_length = 15
        polyorder = 3
        
        pos_smooth = signal.savgol_filter(pos, window_length, polyorder)
        vel_smooth = signal.savgol_filter(pos, window_length, polyorder, deriv=1, delta=dt)
        acc_smooth = signal.savgol_filter(pos, window_length, polyorder, deriv=2, delta=dt)
        
    elif method == 'centered_difference':
        # 中心差分法
        pos_smooth = pos.copy()  # 不进行位置平滑
        
        # 中心差分求速度
        vel_smooth = np.zeros_like(pos)
        vel_smooth[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
        vel_smooth[0] = (pos[1] - pos[0]) / dt
        vel_smooth[-1] = (pos[-1] - pos[-2]) / dt
        
        # 中心差分求加速度
        acc_smooth = np.zeros_like(vel_smooth)
        acc_smooth[1:-1] = (vel_smooth[2:] - vel_smooth[:-2]) / (2 * dt)
        acc_smooth[0] = (vel_smooth[1] - vel_smooth[0]) / dt
        acc_smooth[-1] = (vel_smooth[-1] - vel_smooth[-2]) / dt
    
    return pos_smooth, vel_smooth, acc_smooth

# 使用不同方法计算
methods = ['direct_diff', 'centered_difference', 'butterworth', 'savgol']
fc_pos = 20
fc_vel = 10

results = {}

# 直接差分
results['direct_diff'] = {
    'position': pos_raw,
    'velocity': vel_direct,
    'acceleration': acc_direct
}

# 其他方法
for method in methods[1:]:  # 跳过第一个（直接差分）
    pos_smooth, vel_smooth, acc_smooth = smooth_differentiation(
        pos_raw, dt=dt, fc_pos=fc_pos, fc_vel=fc_vel, method=method
    )
    results[method] = {
        'position': pos_smooth,
        'velocity': vel_smooth,
        'acceleration': acc_smooth
    }

# ============================================
# 图表1：四种方法的速度对比
# ============================================
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# 子图1：速度对比（整体）
zoom_points = min(800, len(t))
for i, method in enumerate(methods):
    if method == 'direct_diff':
        ax1.plot(t[:zoom_points], results[method]['velocity'][:zoom_points], 
                'k-', alpha=0.5, linewidth=1, label='Direct Difference (Raw)')
    else:
        colors = ['b', 'g', 'r']
        ax1.plot(t[:zoom_points], results[method]['velocity'][:zoom_points], 
                colors[i-1], linewidth=1.5, 
                label=method.replace('_', ' ').title())

ax1.set_title(f'Velocity Comparison: Direct Difference vs Smoothing Methods\n(First {zoom_points} points)', 
             fontsize=14, fontweight='bold')
ax1.set_xlabel('Sample Points')
ax1.set_ylabel('Velocity')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best')
ax1.set_xlim([0, zoom_points])

# 子图2：加速度对比（整体）
for i, method in enumerate(methods):
    if method == 'direct_diff':
        ax2.plot(t[:zoom_points], results[method]['acceleration'][:zoom_points], 
                'k-', alpha=0.5, linewidth=1, label='Direct Difference (Raw)')
    else:
        colors = ['b', 'g', 'r']
        ax2.plot(t[:zoom_points], results[method]['acceleration'][:zoom_points], 
                colors[i-1], linewidth=1.5, 
                label=method.replace('_', ' ').title())

ax2.set_title(f'Acceleration Comparison: Direct Difference vs Smoothing Methods\n(First {zoom_points} points)', 
             fontsize=14, fontweight='bold')
ax2.set_xlabel('Sample Points')
ax2.set_ylabel('Acceleration')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best')
ax2.set_xlim([0, zoom_points])

plt.tight_layout()
plt.show()

# ============================================
# 图表2：详细对比（分方法显示）
# ============================================
method_names = {
    'direct_diff': 'Direct Difference',
    'centered_difference': 'Centered Difference',
    'butterworth': 'Butterworth Filter',
    'savgol': 'Savitzky-Golay Filter'
}

fig2, axes = plt.subplots(4, 3, figsize=(18, 12))
fig2.suptitle(f'Detailed Comparison of Different Differentiation Methods\n(fs={fs}Hz, fc_pos={fc_pos}Hz, fc_vel={fc_vel}Hz)', 
              fontsize=16, fontweight='bold', y=1.02)

display_points = min(500, len(t))

for row, method in enumerate(methods):
    # 位置对比
    ax_pos = axes[row, 0]
    if method == 'direct_diff':
        ax_pos.plot(t[:display_points], pos_raw[:display_points], 
                   'k-', alpha=0.5, linewidth=0.8)
    else:
        ax_pos.plot(t[:display_points], pos_raw[:display_points], 
                   'k-', alpha=0.3, linewidth=0.5, label='Raw')
        ax_pos.plot(t[:display_points], results[method]['position'][:display_points], 
                   'b-', linewidth=1.2, label='Smoothed')
    ax_pos.set_title(f'{method_names[method]}\nPosition', fontsize=10, fontweight='bold')
    ax_pos.set_ylabel('Position')
    ax_pos.grid(True, alpha=0.3)
    if row == 0:
        ax_pos.legend(fontsize=8)
    
    # 速度
    ax_vel = axes[row, 1]
    ax_vel.plot(t[:display_points], results[method]['velocity'][:display_points], 
               'g-', linewidth=1.5)
    ax_vel.set_title(f'Velocity', fontsize=10, fontweight='bold')
    ax_vel.set_ylabel('Velocity')
    ax_vel.grid(True, alpha=0.3)
    
    # 加速度
    ax_acc = axes[row, 2]
    ax_acc.plot(t[:display_points], results[method]['acceleration'][:display_points], 
               'r-', linewidth=1.5)
    ax_acc.set_title(f'Acceleration', fontsize=10, fontweight='bold')
    ax_acc.set_ylabel('Acceleration')
    ax_acc.grid(True, alpha=0.3)
    
    # 只在最后一行显示x轴标签
    if row == 3:
        ax_pos.set_xlabel('Sample Points')
        ax_vel.set_xlabel('Sample Points')
        ax_acc.set_xlabel('Sample Points')
    else:
        ax_pos.set_xticklabels([])
        ax_vel.set_xticklabels([])
        ax_acc.set_xticklabels([])

plt.tight_layout()
plt.show()

# ============================================
# 图表3：统计对比
# ============================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 速度统计
vel_means = []
vel_stds = []
acc_means = []
acc_stds = []

for method in methods:
    vel = results[method]['velocity']
    acc = results[method]['acceleration']
    
    vel_means.append(np.mean(vel))
    vel_stds.append(np.std(vel))
    acc_means.append(np.mean(acc))
    acc_stds.append(np.std(acc))

x = np.arange(len(methods))
width = 0.35

# 速度标准差对比
bars1 = ax1.bar(x - width/2, vel_stds, width, label='Velocity STD', 
               color='green', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Method')
ax1.set_ylabel('Standard Deviation')
ax1.set_title('Velocity Standard Deviation Comparison\n(Lower is smoother)', 
             fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([method_names[m] for m in methods], rotation=15, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars1, vel_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(vel_stds),
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# 加速度标准差对比
bars2 = ax2.bar(x - width/2, acc_stds, width, label='Acceleration STD', 
               color='red', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Method')
ax2.set_ylabel('Standard Deviation')
ax2.set_title('Acceleration Standard Deviation Comparison\n(Lower is smoother)', 
             fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([method_names[m] for m in methods], rotation=15, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars2, acc_stds):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(acc_stds),
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================
# 图表4：直方图对比
# ============================================
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# 速度分布直方图
axes[0, 0].hist(vel_direct, bins=100, alpha=0.7, color='gray', 
               label='Direct Diff', density=True, edgecolor='black')
axes[0, 0].set_title('Velocity Distribution: Direct Difference', 
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Velocity')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

colors = ['blue', 'green', 'red']
for idx, method in enumerate(methods[1:], 1):
    axes[0, 1].hist(results[method]['velocity'], bins=100, alpha=0.5, 
                   color=colors[idx-1], label=method_names[method], 
                   density=True, edgecolor='black')
axes[0, 1].set_title('Velocity Distribution: Smoothing Methods', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Velocity')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 加速度分布直方图
axes[1, 0].hist(acc_direct, bins=100, alpha=0.7, color='gray', 
               label='Direct Diff', density=True, edgecolor='black')
axes[1, 0].set_title('Acceleration Distribution: Direct Difference', 
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Acceleration')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

for idx, method in enumerate(methods[1:], 1):
    axes[1, 1].hist(results[method]['acceleration'], bins=100, alpha=0.5, 
                   color=colors[idx-1], label=method_names[method], 
                   density=True, edgecolor='black')
axes[1, 1].set_title('Acceleration Distribution: Smoothing Methods', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Acceleration')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 保存结果
# ============================================
result_df = pd.DataFrame({
    'time': t * dt,
    'position_raw': pos_raw,
    'velocity_direct': vel_direct,
    'acceleration_direct': acc_direct,
})

for method in methods[1:]:
    result_df[f'position_{method}'] = results[method]['position']
    result_df[f'velocity_{method}'] = results[method]['velocity']
    result_df[f'acceleration_{method}'] = results[method]['acceleration']

result_df.to_csv('all_methods_comparison.csv', index=False)
print(f"\nResults saved to: all_methods_comparison.csv")

# ============================================
# 详细统计信息
# ============================================
print("\n" + "="*80)
print("COMPARISON OF DIFFERENTIATION METHODS")
print("="*80)
print(f"Sampling frequency: {fs} Hz, Cutoff frequency: {fc_pos}/{fc_vel} Hz")
print(f"Total data points: {len(pos_raw)}")
print("\n" + "-"*80)
print("STATISTICAL ANALYSIS")
print("-"*80)

for method in methods:
    vel = results[method]['velocity']
    acc = results[method]['acceleration']
    
    print(f"\n【{method_names[method]}】")
    print(f"  Velocity Statistics:")
    print(f"    Mean: {np.mean(vel):.6f}")
    print(f"    STD:  {np.std(vel):.6f}")
    print(f"    Min:  {np.min(vel):.6f}")
    print(f"    Max:  {np.max(vel):.6f}")
    print(f"    Range:{np.max(vel)-np.min(vel):.6f}")
    
    print(f"  Acceleration Statistics:")
    print(f"    Mean: {np.mean(acc):.6f}")
    print(f"    STD:  {np.std(acc):.6f}")
    print(f"    Min:  {np.min(acc):.6f}")
    print(f"    Max:  {np.max(acc):.6f}")
    print(f"    Range:{np.max(acc)-np.min(acc):.6f}")

# 平滑度改善百分比
print("\n" + "-"*80)
print("IMPROVEMENT OVER DIRECT DIFFERENCE")
print("-"*80)
direct_vel_std = np.std(vel_direct)
direct_acc_std = np.std(acc_direct)

for method in methods[1:]:
    vel_std = np.std(results[method]['velocity'])
    acc_std = np.std(results[method]['acceleration'])
    
    vel_improvement = (1 - vel_std/direct_vel_std) * 100
    acc_improvement = (1 - acc_std/direct_acc_std) * 100
    
    print(f"\n{method_names[method]}:")
    print(f"  Velocity smoothness improvement: {vel_improvement:.1f}%")
    print(f"  Acceleration smoothness improvement: {acc_improvement:.1f}%")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("1. Direct Difference: Most noisy, not recommended for control systems")
print("2. Centered Difference: Better than direct difference, simple")
print("3. Butterworth Filter: Good balance of smoothness and responsiveness")
print("4. Savitzky-Golay: Best for preserving peaks while smoothing")
print("\nFor motor control applications, Butterworth Filter is recommended.")
print(f"Suggested cutoff: fc_pos = {0.1*fs:.0f} Hz, fc_vel = {0.05*fs:.0f} Hz")