import pyrealsense2 as rs
import numpy as np
import time

# 初始化 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()

# 启用 IMU 数据流
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # 加速度计
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)   # 陀螺仪

# 启动管道
pipeline.start(config)

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames()

        # 获取加速度计数据
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            # print(f"Accelerometer: X={accel_data.x:.3f}, Y={accel_data.y:.3f}, Z={accel_data.z:.3f} m/s²")

        # 获取陀螺仪数据
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            print(f"Gyroscope: X={gyro_data.x:.3f}, Y={gyro_data.y:.3f}, Z={gyro_data.z:.3f} rad/s")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("Stopping pipeline...")
finally:
    # 停止管道
    pipeline.stop()