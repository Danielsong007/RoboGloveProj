import pyrealsense2 as rs
import numpy as np
import time
from ahrs.filters import Madgwick

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 启用 IMU 数据流
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # 加速度计
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # 陀螺仪

# 启动管道
pipeline.start(config)

# 初始化 Madgwick 滤波器
madgwick = Madgwick()

# 初始化四元数
q = np.array([1.0, 0.0, 0.0, 0.0])  # 初始四元数 (w, x, y, z)

# 初始化时间变量
previous_time = None



# 四元数转欧拉角函数
def quaternion_to_euler(q):
    w, x, y, z = q
    # 计算横滚角 (roll)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # 计算俯仰角 (pitch)
    pitch = np.arcsin(2 * (w * y - z * x))
    # 计算偏航角 (yaw)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    # 转换为角度
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    return roll, pitch, yaw


try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()

        # 获取当前时间
        current_time = frames.get_timestamp()

        # 计算时间间隔
        if previous_time is not None:
            dt = (current_time - previous_time) / 1000.0  # 转换为秒
        else:
            dt = 1.0 / 200  # 初始时间间隔（假设帧率为 200 Hz）
        previous_time = current_time

        # 获取加速度计数据
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            ax, ay, az = accel_data.x, accel_data.y, accel_data.z

        # 获取陀螺仪数据
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            gx, gy, gz = gyro_data.x, gyro_data.y, gyro_data.z

            # 使用 Madgwick 滤波器更新四元数
            q = madgwick.updateIMU(q, gyr=np.array([gx, gy, gz]), acc=np.array([ax, ay, az]))

            # 将四元数转换为欧拉角
            roll, pitch, yaw = quaternion_to_euler(q)

            # 打印角度信息
            print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
            time.sleep(0.2)

except KeyboardInterrupt:
    print("Stopping pipeline...")
finally:
    # 停止管道
    pipeline.stop()
