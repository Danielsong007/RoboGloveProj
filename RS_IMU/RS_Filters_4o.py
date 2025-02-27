import pyrealsense2 as rs
import math
import time

class ComplementaryFilter:
    """互补滤波器实现"""
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.angle = 0.0  # 滤波后的角度

    def update(self, gyro_rate, dt, accel_angle):
        """
        更新互补滤波器的角度
        :param gyro_rate: 陀螺仪角速度 (deg/s)
        :param dt: 时间间隔 (s)
        :param accel_angle: 从加速度计计算的角度 (deg)
        """
        gyro_angle = self.angle + gyro_rate * dt  # 陀螺仪角度积分
        self.angle = self.alpha * gyro_angle + (1 - self.alpha) * accel_angle
        return self.angle

class KalmanFilter:
    """卡尔曼滤波器实现"""
    def __init__(self):
        self.angle = 0.0  # 滤波后的角度
        self.bias = 0.0  # 偏差
        self.P = [[0, 0], [0, 0]]  # 误差协方差矩阵
        self.Q_angle = 0.001  # 角度过程噪声
        self.Q_bias = 0.003  # 偏差过程噪声
        self.R_measure = 0.03  # 测量噪声

    def update(self, accel_angle, gyro_rate, dt):
        """
        更新卡尔曼滤波器的角度
        :param accel_angle: 从加速度计计算的角度 (deg)
        :param gyro_rate: 陀螺仪角速度 (deg/s)
        :param dt: 时间间隔 (s)
        """
        # 预测
        rate = gyro_rate - self.bias
        self.angle += dt * rate

        self.P[0][0] += dt * (dt * self.P[1][1] - self.P[0][1] - self.P[1][0] + self.Q_angle)
        self.P[0][1] -= dt * self.P[1][1]
        self.P[1][0] -= dt * self.P[1][1]
        self.P[1][1] += self.Q_bias * dt

        # 更新
        S = self.P[0][0] + self.R_measure
        K = [self.P[0][0] / S, self.P[1][0] / S]
        y = accel_angle - self.angle
        self.angle += K[0] * y
        self.bias += K[1] * y

        P00_temp = self.P[0][0]
        P01_temp = self.P[0][1]

        self.P[0][0] -= K[0] * P00_temp
        self.P[0][1] -= K[0] * P01_temp
        self.P[1][0] -= K[1] * P00_temp
        self.P[1][1] -= K[1] * P01_temp

        return self.angle

def calculate_angles_from_accel(accel):
    """根据加速度计数据计算俯仰角（Pitch）和横滚角（Roll）"""
    ax, ay, az = accel
    pitch = math.atan2(ax, math.sqrt(ay ** 2 + az ** 2)) * 180 / math.pi
    roll = math.atan2(ay, math.sqrt(ax ** 2 + az ** 2)) * 180 / math.pi
    return pitch, roll

def main():
    # 初始化 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用 IMU 流（加速度计和陀螺仪）
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    pipeline.start(config)

    # 初始化滤波器
    complementary_filter = ComplementaryFilter(alpha=0.98)
    kalman_filter = KalmanFilter()

    print("请选择滤波算法：")
    print("1. 互补滤波")
    print("2. 卡尔曼滤波")
    algorithm = int(input("请输入算法编号（1 或 2）："))

    if algorithm not in [1, 2]:
        print("无效选择，默认使用互补滤波。")
        algorithm = 1

    print("正在获取角度信息，按 Ctrl+C 停止程序...")

    try:
        prev_time = time.time()

        while True:
            # 等待 IMU 帧
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                # 获取加速度计数据
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                accel = (accel_data.x, accel_data.y, accel_data.z)

                # 获取陀螺仪数据
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                gyro = (gyro_data.x, gyro_data.y, gyro_data.z)

                # 从加速度计计算角度
                accel_pitch, _ = calculate_angles_from_accel(accel)

                # 从陀螺仪获取角速度（单位转换为度/秒）
                gyro_pitch_rate = gyro[0] * 180 / math.pi

                # 计算时间间隔
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time

                # 根据选择的算法计算角度
                if algorithm == 1:
                    # 互补滤波
                    fused_angle = complementary_filter.update(gyro_pitch_rate, dt, accel_pitch)
                elif algorithm == 2:
                    # 卡尔曼滤波
                    fused_angle = kalman_filter.update(accel_pitch, gyro_pitch_rate, dt)

                # 打印实时角度信息
                print(f"Fused Angle (Pitch): {fused_angle:.2f}°")

            # 限制刷新率
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n程序已停止。")
    finally:
        # 停止 RealSense 管道
        pipeline.stop()

if __name__ == "__main__":
    main()