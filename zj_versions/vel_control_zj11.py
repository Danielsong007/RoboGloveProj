import time
import threading
import socket
import numpy as np
from collections import deque
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor

# =========================
# 主控制系统类
# =========================
class XYZFollowControlSystem:
    def __init__(self, rope_port='/dev/ttyUSB0', touch_port=65432):
        # 状态变量
        self.Rope_S = 0          # 拉力传感器值
        self.Touch_S = 0         # 触觉传感器值
        self.Ave_Rope_S = 0
        self.Ave_Touch_S = 0
        self.prev_Rope_S = 0     # 上一次拉力值，用于计算变化率
        self.mode = "IDLE"       # 状态机：IDLE, FOLLOW, SAFETY
        self.Touch_valve = 200   # 触觉阈值，判断是否搬运
        self.force_threshold = 5 # 拉力变化阈值，判断手是否移动
        self.max_speed = 4000    # 电机最大速度
        self.k = 50              # 跟随比例系数（调节灵敏度）

        # 防抖计时器
        self.touch_start = None
        self.release_start = None

        # 控制器
        self.xyz = xyz_utils()

        # 传感器
        self.rope_sensor = DaYangSensor(rope_port, 0)
        self.touch_port = touch_port

        # 缓冲区
        self.rope_buffer = deque(maxlen=3)
        self.touch_buffer = deque(maxlen=3)

    # =========================
    # 传感器线程
    # =========================
    def read_rope_sensor(self):
        while True:
            self.Rope_S = self.rope_sensor.read_angles()  # 实际是拉力值
            self.rope_buffer.append(self.Rope_S)
            self.Ave_Rope_S = np.mean(self.rope_buffer)
            time.sleep(0.001)

    def touch_sensor_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.touch_port))
            s.listen()
            print(f"Touch sensor server started on port {self.touch_port}")
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    msg = data.decode()
                    self.Touch_S = int(msg.split()[-1])
                    self.touch_buffer.append(self.Touch_S)
                    self.Ave_Touch_S = np.mean(self.touch_buffer)

    # =========================
    # 状态机逻辑（带防抖）
    # =========================
    def update_state(self):
        now = time.time()
        if self.mode == "IDLE":
            if self.Ave_Touch_S > self.Touch_valve:
                if self.touch_start is None:
                    self.touch_start = now
                elif now - self.touch_start > 0.3:  # 连续0.3秒超过阈值
                    self.mode = "FOLLOW"
                    print("进入跟随模式")
                    self.touch_start = None
            else:
                self.touch_start = None

        elif self.mode == "FOLLOW":
            if self.Ave_Touch_S < self.Touch_valve:
                if self.release_start is None:
                    self.release_start = now
                elif now - self.release_start > 0.3:  # 连续0.3秒低于阈值
                    self.mode = "SAFETY"
                    print("手松开，进入安全模式")
                    self.release_start = None
            else:
                self.release_start = None

        elif self.mode == "SAFETY":
            if self.Ave_Touch_S > self.Touch_valve:
                self.mode = "FOLLOW"
                print("重新进入跟随模式")

    # =========================
    # 跟随控制逻辑
    # =========================
    def follow_control(self):
        delta_force = self.Rope_S - self.prev_Rope_S
        self.prev_Rope_S = self.Rope_S

        if abs(delta_force) < self.force_threshold:
            motor_speed = 0  # 手静止，电机保持
        else:
            motor_speed = self.k * delta_force  # 根据拉力变化率调整速度

        motor_speed = np.clip(motor_speed, -self.max_speed, self.max_speed)
        self.xyz.AxisMode_Jog(3, 30, motor_speed)

    # =========================
    # 安全模式逻辑
    # =========================
    def safety_mode(self):
        print("安全模式：缓慢下降")
        self.xyz.AxisMode_Jog(3, 30, -500)  # 缓慢下降
        time.sleep(0.05)

    # =========================
    # 主控制循环
    # =========================
    def control_loop(self):
        self.xyz.OpenEnableZero_ALL()
        InitPos = self.xyz.Safe_Jog()
        print("系统初始化完成，等待触觉数据...")
        while self.Touch_S == 0:
            time.sleep(0.1)

        while True:
            self.update_state()
            if self.mode == "FOLLOW":
                self.follow_control()
            elif self.mode == "SAFETY":
                self.safety_mode()
            elif self.mode == "IDLE":
                self.xyz.AxisMode_Jog(3, 30, 0)  # 停止

            time.sleep(0.01)

    # =========================
    # 启动系统
    # =========================
    def start(self):
        threading.Thread(target=self.read_rope_sensor, daemon=True).start()
        threading.Thread(target=self.touch_sensor_server, daemon=True).start()
        self.control_loop()

# =========================
# 主入口
# =========================
if __name__ == "__main__":
    system = XYZFollowControlSystem()
    system.start()