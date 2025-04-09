import pybullet as p
import pybullet_data
import os
import time
import numpy as np

# 初始化物理引擎
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# 加载自定义URDF模型
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "pendulum.urdf")
pendulum = p.loadURDF(urdf_path, [0, 0, 2], useFixedBase=True)

# 获取关节索引
joint_indices = {}
for i in range(p.getNumJoints(pendulum)):
    joint_info = p.getJointInfo(pendulum, i)
    joint_name = joint_info[1].decode("utf-8")
    joint_indices[joint_name] = i

# 打印关节信息
print("可用关节:", joint_indices)

# 设置关节控制模式（允许自由摆动）
p.setJointMotorControl2(
    pendulum, 
    joint_indices["joint_x_axis"], 
    p.VELOCITY_CONTROL, 
    targetVelocity=0, 
    force=0
)
p.setJointMotorControl2(
    pendulum, 
    joint_indices["joint_y_axis"], 
    p.VELOCITY_CONTROL, 
    targetVelocity=0, 
    force=0
)

# 设置初始扰动（模拟释放动作）
p.resetJointState(
    pendulum, 
    joint_indices["joint_x_axis"], 
    targetValue=np.radians(30),  # 初始X轴角度30°
    targetVelocity=1.0           # 初始角速度1 rad/s
)
p.resetJointState(
    pendulum, 
    joint_indices["joint_y_axis"], 
    targetValue=np.radians(15),  # 初始Y轴角度15°
    targetVelocity=0.5           # 初始角速度0.5 rad/s
)

# 仿真循环
while True:
    # 获取当前状态
    x_pos, x_vel, _, _ = p.getJointState(pendulum, joint_indices["joint_x_axis"])
    y_pos, y_vel, _, _ = p.getJointState(pendulum, joint_indices["joint_y_axis"])
    
    print(
        f"X轴角度: {np.degrees(x_pos):.1f}°, "
        f"Y轴角度: {np.degrees(y_pos):.1f}° | "
        f"X速度: {x_vel:.2f}, Y速度: {y_vel:.2f} rad/s"
    )
    
    p.stepSimulation()
    time.sleep(1.0/240.0)  # 240Hz仿真频率

p.disconnect()