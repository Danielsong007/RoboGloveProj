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
    urdf_path = os.path.join(os.path.dirname(__file__), "wholeR.urdf")
    robot = p.loadURDF(urdf_path, [0, 0, 3], useFixedBase=True)
    joint_indices = {p.getJointInfo(robot, i)[1].decode():i 
                    for i in range(p.getNumJoints(robot))}

    # 将所有关节初始状态设为0
    for joint_name, joint_idx in joint_indices.items():
        p.resetJointState(robot,joint_idx,targetValue=0,targetVelocity=0)
        p.setJointMotorControl2(robot,joint_idx,p.VELOCITY_CONTROL,targetVelocity=0,force=0)

    try:
        t = 0
        while True:
            t += 1/240.
            targets = [
                0.8 * np.sin(2*np.pi*0.6*t),      # X轴
                0. * np.cos(2*np.pi*0.2*t),       # Y轴
                0. * np.cos(2*np.pi*0.2*t) + 2.5  # Z轴
            ]
            p.setJointMotorControlArray(robot,
                [joint_indices["x_joint"], joint_indices["y_joint"], joint_indices["z_joint"]],
                p.POSITION_CONTROL, targetPositions=targets, forces=[50, 50, 90])
            p.stepSimulation()
            time.sleep(1/240)
            
    except KeyboardInterrupt:
        p.disconnect()

if __name__ == "__main__":
    main()

