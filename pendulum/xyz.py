import os
import pybullet as p
import pybullet_data
import time
import numpy as np

def main():
    # 初始化物理引擎
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # 加载机器人
    urdf_path = os.path.join(os.path.dirname(__file__), "xyz_platform.urdf")
    robot = p.loadURDF(urdf_path, [0, 0, 3], useFixedBase=True)
    joint_indices = {p.getJointInfo(robot, i)[1].decode():i 
                    for i in range(p.getNumJoints(robot))}
    
    # 主控制循环
    try:
        t = 0
        while True:
            t += 1/240.
            targets = [
                0.8 * np.sin(2*np.pi*0.2*t),      # X轴
                0.8 * np.cos(2*np.pi*0.2*t),      # Y轴
                2.5  # Z轴
            ]
            p.setJointMotorControlArray(
                robot,
                [joint_indices["x_joint"], joint_indices["y_joint"], joint_indices["z_joint"]],
                p.POSITION_CONTROL,
                targetPositions=targets,
                forces=[500, 500, 800]  # Z轴增加驱动力
            )
            p.stepSimulation()
            time.sleep(1/240.)
            
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()

