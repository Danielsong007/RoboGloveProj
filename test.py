import pybullet as p
import pybullet_data
import time
import numpy as np

# 启动 PyBullet
physicsClient = p.connect(p.GUI)  # 使用图形界面
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 添加 PyBullet 数据路径
p.setGravity(0, 0, -9.81)  # 设置重力

# 创建地面
planeId = p.loadURDF("plane.urdf")

# 创建绳子
rope_length = 2.0  # 绳子长度
rope_mass = 0.1  # 绳子质量
rope_radius = 0.02  # 绳子半径
rope_collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=rope_radius, height=rope_length)
rope_visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=rope_radius, length=rope_length, rgbaColor=[0.5, 0.5, 0.5, 1])
rope_position = [0, 0, rope_length / 2]  # 绳子的初始位置
rope_orientation = p.getQuaternionFromEuler([0, 0, 0])
rope = p.createMultiBody(rope_mass, rope_collision_shape, rope_visual_shape, rope_position, rope_orientation)

# 创建钢球
ball_radius = 0.1  # 钢球半径
ball_mass = 1.0  # 钢球质量
ball_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
ball_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0.8, 0.2, 0.2, 1])
ball_position = [0, 0, -rope_length / 2]  # 钢球的初始位置
ball_orientation = p.getQuaternionFromEuler([0, 0, 0])
ball = p.createMultiBody(ball_mass, ball_collision_shape, ball_visual_shape, ball_position, ball_orientation)

# 创建悬挂点
anchor_position = [0, 0, rope_length]  # 悬挂点的初始位置
anchor_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.2, 0.8, 0.2, 1])
anchor = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=anchor_visual_shape, basePosition=anchor_position)

# 创建绳子与悬挂点、钢球的连接
p.createConstraint(anchor, -1, rope, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, -rope_length / 2], [0, 0, 0])
p.createConstraint(rope, -1, ball, -1, p.JOINT_FIXED, [0, 0, -rope_length / 2], [0, 0, 0], [0, 0, 0])

# 设置悬挂点的周期性运动
amplitude = 1.0  # 运动幅度
frequency = 1.0  # 运动频率
start_time = time.time()

# 仿真循环
while True:
    # 计算悬挂点的位置
    t = time.time() - start_time
    x = amplitude * np.sin(2 * np.pi * frequency * t)
    anchor_position = [x, 0, rope_length]
    
    # 更新悬挂点的位置
    p.resetBasePositionAndOrientation(anchor, anchor_position, [0, 0, 0, 1])
    
    # 仿真步进
    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # 控制仿真速度