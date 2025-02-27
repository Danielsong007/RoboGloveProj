import pyrealsense2 as rs
import math
import time

def calculate_angles_from_accel(accel):
    """
    根据加速度计数据计算俯仰角（Pitch）和横滚角（Roll）。
    假设加速度计的单位是 m/s²。
    """
    # 从加速度计中获取 x, y, z 分量
    ax, ay, az = accel

    # 计算俯仰角（Pitch）和横滚角（Roll），角度单位为度
    pitch = math.atan2(ax, math.sqrt(ay ** 2 + az ** 2)) * 180 / math.pi
    roll = math.atan2(ay, math.sqrt(ax ** 2 + az ** 2)) * 180 / math.pi

    return pitch, roll

def main():
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
            # 等待 IMU 帧
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if accel_frame:
                # 获取加速度计数据
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                accel = (accel_data.x, accel_data.y, accel_data.z)

                # 根据加速度计数据计算角度
                pitch, roll = calculate_angles_from_accel(accel)

                # 打印实时角度信息
                print(f"Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")

            # 为了避免过高的数据刷新率，可以加入一些延时（可选）
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n程序已停止。")
    finally:
        # 停止 RealSense 管道
        pipeline.stop()

if __name__ == "__main__":
    main()