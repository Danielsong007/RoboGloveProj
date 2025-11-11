import time
import threading
import socket
import numpy as np
from collections import deque
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor

# =========================
# PID 控制器类
# =========================
class PIDController:
    def __init__(self, kp=0.015, ki=0.0001, kd=0.001, max_output=4000):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.prev_error = 0
        self.max_output = max_output

    def compute(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, -self.max_output, self.max_output)

# =========================
# 主控制系统类
# =========================
class XYZControlSystem:
    def __init__(self, rope_port='/dev/ttyUSB0', touch_port=65432):
        # 状态变量
        self.Rope_S = 0
        self.Touch_S = 0
        self.Ave_Rope_S = 0
        self.Ave_Touch_S = 0
        self.Weight = 0
        self.Touch_valve = 70
        self.mode = "IDLE"  # 状态机：IDLE, MEASURING, LOADING
        self.measurement_done = False

        # 防抖计时器
        self.touch_start = None
        self.release_start = None

        # 控制器
        self.xyz = xyz_utils()
        self.pid = PIDController()

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
            self.Rope_S = self.rope_sensor.read_angles()
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
                elif now - self.touch_start > 0.5:  # 连续0.5秒超过阈值
                    self.mode = "MEASURING"
                    self.touch_start = None
            else:
                self.touch_start = None

        elif self.mode == "MEASURING":
            if self.measurement_done:
                self.mode = "LOADING"

        elif self.mode == "LOADING":
            if self.Ave_Touch_S < self.Touch_valve:
                if self.release_start is None:
                    self.release_start = now
                elif now - self.release_start > 0.5:  # 连续0.5秒低于阈值
                    print("物体被移除，重置重量")
                    self.Weight = 0
                    self.measurement_done = False
                    self.mode = "IDLE"
                    self.release_start = None
            else:
                self.release_start = None

    # =========================
    # 测量重量逻辑
    # =========================
    def measure_weight(self):
        print("开始测量重量...")
        cur_time = time.time()
        while self.Rope_S < 500 and (time.time() - cur_time) < 1:
            self.xyz.AxisMode_Jog(3, 30, 2000)
            time.sleep(0.01)
        if self.Rope_S > 500:
            samples_rope = []
            samples_touch = []
            for _ in range(50):
                samples_rope.append(self.Rope_S)
                samples_touch.append(self.Touch_S)
                time.sleep(0.01)
            self.Weight = np.mean(samples_rope) + 0.2 * np.mean(samples_touch)
            print(f"测量完成，重量: {self.Weight}")
            self.measurement_done = True
        else:
            print("测量失败：超时")
        self.xyz.AxisMode_Jog(3, 30, 0)

    # =========================
    # 主控制循环
    # =========================
    def control_loop(self):
        self.xyz.OpenEnableZero_ALL()
        InitPos = self.xyz.Safe_Jog()
        print("系统初始化完成，等待触摸数据...")
        while self.Touch_S == 0:
            time.sleep(0.1)

        while True:
            self.update_state()
            if self.mode == "MEASURING" and not self.measurement_done:
                self.measure_weight()
            elif self.mode == "LOADING":
                velocity = self.pid.compute(self.Weight, self.Ave_Rope_S)
                self.xyz.AxisMode_Jog(3, 30, velocity)
            elif self.mode == "IDLE":
                velocity = self.pid.compute(40, self.Ave_Rope_S)
                self.xyz.AxisMode_Jog(3, 30, velocity)

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
    system = XYZControlSystem()
    system.start()