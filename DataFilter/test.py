import pybullet as p
import pybullet_data
import numpy as np
import time

# 初始化
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)

# 创建地面
plane_id = p.loadURDF("plane.urdf")

# 创建箱子
box_start_pos = [0, 0, 1.0]
box_id = p.loadURDF("cube.urdf", basePosition=box_start_pos, globalScaling=0.2)
p.changeDynamics(box_id, -1, mass=2.0)  # 设置质量为2kg

# 创建"电机"（实际上用一个固定的点表示）
motor_pos = [0, 0, 2.0]
motor_sphere = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1]),
    basePosition=motor_pos
)

# 用一根弹簧模拟绳子（简化）
rope_spring_stiffness = 100.0
rope_damping = 1.0

# PID控制器参数
desired_force = 5.0
Kp = 30.0
Ki = 5.0
Kd = 2.0
integral = 0.0
prev_error = 0.0

# 手的位置干扰
def hand_force(t):
    return (3.0 * np.sin(0.5 * 2*np.pi*t) +
            2.0 * np.sin(2.0 * 2*np.pi*t) +
            1.0 * np.sin(5.0 * 2*np.pi*t))

# 仿真循环
sim_time = 0
sim_times = []
measured_forces = []
control_signals = []

print("力控制仿真开始...")
print("期望力:", desired_force, "N")

while sim_time < 10.0:
    # 模拟手施加的干扰力
    h_force = hand_force(sim_time)
    
    # 在箱子上施加手的作用力
    p.applyExternalForce(
        objectUniqueId=box_id,
        linkIndex=-1,
        forceObj=[0, 0, h_force],
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME
    )
    
    # 获取箱子状态
    box_pos, _ = p.getBasePositionAndOrientation(box_id)
    box_vel, _ = p.getBaseVelocity(box_id)
    
    # 绳子弹簧力（电机和箱子之间的弹簧）
    spring_force_z = rope_spring_stiffness * (motor_pos[2] - box_pos[2])
    
    # 计算传感器力（简化：假设传感器测量手施加的力）
    # 实际上应该是综合力，这里简化处理
    measured_force = h_force
    
    # PID控制
    error = desired_force - measured_force
    integral += error * (1/240)
    derivative = (error - prev_error) / (1/240) if prev_error is not None else 0
    control = Kp * error + Ki * integral + Kd * derivative
    prev_error = error
    
    # 限制控制信号
    control = np.clip(control, -50, 50)
    
    # 应用电机控制力（通过移动"电机"位置来影响弹簧力）
    # 实际上，我们应该通过改变motor_pos来影响绳子张力
    # 简化：直接对箱子施加控制力
    p.applyExternalForce(
        objectUniqueId=box_id,
        linkIndex=-1,
        forceObj=[0, 0, control],
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME
    )
    
    # 存储数据
    sim_times.append(sim_time)
    measured_forces.append(measured_force)
    control_signals.append(control)
    
    # 执行仿真步
    p.stepSimulation()
    time.sleep(1/240)
    sim_time += 1/240
    
    # 在仿真中绘制线（绳子）
    p.addUserDebugLine(motor_pos, box_pos, [1, 1, 0], 2, 1/240)

# 断开连接
p.disconnect()

# 绘制结果
import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# 1. 力跟踪
ax1.plot(sim_times, measured_forces, 'b-', label='测量力')
ax1.axhline(desired_force, color='r', linestyle='--', label='期望力')
ax1.set_ylabel('力 (N)')
ax1.set_title('力跟踪控制')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 力误差
errors = desired_force - np.array(measured_forces)
ax2.plot(sim_times, errors, 'r-')
ax2.axhline(0, color='k', linestyle='--')
ax2.set_ylabel('误差 (N)')
ax2.set_title('力跟踪误差')
ax2.grid(True, alpha=0.3)

# 3. 控制信号
ax3.plot(sim_times, control_signals, 'g-')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('控制信号')
ax3.set_title('电机控制信号')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印统计
print("\n仿真结果:")
print("="*40)
print(f"最大误差: {np.max(np.abs(errors)):.3f} N")
print(f"RMS误差: {np.sqrt(np.mean(errors**2)):.3f} N")
print(f"平均误差: {np.mean(errors):.3f} N")
print("="*40)