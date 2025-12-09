import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('DataFilter/sensor_data.csv')
pos_raw = df.iloc[:, 4].astype(float).values
t = np.arange(len(pos_raw))

# 2. Parameters
fs = 1000
dt = 1/fs
fc_pos, fc_vel = 20, 10

# 3. Direct Difference Method
vel_raw = np.gradient(pos_raw, dt)
acc_raw = np.gradient(vel_raw, dt)

# 4. Butterworth Filter
def butterworth_filter(pos, dt, fc):
    b, a = signal.butter(2, fc/(1/(2*dt)), 'low')
    return signal.filtfilt(b, a, pos)

pos_smooth = butterworth_filter(pos_raw, dt, fc_pos)
vel_smooth = np.gradient(pos_smooth, dt)
vel_smooth = butterworth_filter(vel_smooth, dt, fc_vel)
acc_smooth = np.gradient(vel_smooth, dt)
# acc_smooth = butterworth_filter(acc_smooth, dt, fc_vel)
# 将加速度除以100000
acc_smooth = acc_smooth / 7500000 * -1

df['Position_Smooth'] = pos_smooth
df['Velocity_Smooth'] = vel_smooth
df['Acceleration_Smooth'] = acc_smooth
df.to_csv('DataFilter/sensor_data.csv', index=False)

# 5. Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Position
ax1.plot(t, pos_raw, 'k-', alpha=0.3, linewidth=1.2, label='Raw Position')
ax1.plot(t, pos_smooth, 'b-', linewidth=1.5, label=f'Filtered Position (fc={fc_pos}Hz)')
ax1.set_ylabel('Position')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Position Comparison')

# Velocity
ax2.plot(t, vel_raw, 'darkred', alpha=0.7, linewidth=1.2, label='Direct Difference')
ax2.plot(t, vel_smooth, 'g-', linewidth=1.5, label=f'Filtered Velocity (fc={fc_vel}Hz)')
ax2.set_ylabel('Velocity')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Velocity Comparison')

# Acceleration
# ax3.plot(t, acc_raw, 'darkred', alpha=0.7, linewidth=1.2, label='Direct Difference')
ax3.plot(t, acc_smooth, 'r-', linewidth=1.5, label=f'Filtered Acceleration (fc={fc_vel}Hz)')
ax3.set_xlabel('Sample Points')
ax3.set_ylabel('Acceleration')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Acceleration Comparison')

plt.suptitle(f'Encoder Data Smoothing Comparison (fs={fs}Hz)', fontweight='bold')
plt.tight_layout()
plt.show()
