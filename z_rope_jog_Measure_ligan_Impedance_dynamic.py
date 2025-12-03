import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading
import socket
import csv


Rope_S = 0
Touch_S = 0
cur_pos_abs = 0
cur_vel = 0
cur_acc = 0

buffer_dyn_Srope = deque([0.0]*3, maxlen=3)
buffer_weight_Srope = deque([0.0]*30, maxlen=50)
buffer_dyn_Stouch = deque([0.0]*3, maxlen=3)
buffer_weight_Stouch = deque([0.0]*30, maxlen=50)
buffer_rising_CurPos = deque([0]*5, maxlen=5)
rising_slope = 0

def read_cur_pos(myXYZ, InitPos):
    global cur_pos_abs
    global cur_vel
    global cur_acc
    global buffer_rising_CurPos
    global rising_slope
    last_time = 0
    last_pos = 0
    last_vel = 0
    while True:
        time.sleep(0.001)
        cur_pos_abs = myXYZ.Get_Pos(3)+InitPos
        cur_time = time.time()
        dt = cur_time -last_time
        cur_vel = (cur_pos_abs - last_pos) / dt / 100000
        cur_acc = (cur_vel - last_vel) / dt / 100
        last_time = cur_time
        last_pos = cur_pos_abs
        last_vel = cur_vel
        buffer_rising_CurPos.append(cur_pos_abs)
        x = np.arange(len(buffer_rising_CurPos))*1000
        rising_slope, _ = np.polyfit(x, -np.array(buffer_rising_CurPos), 1)

def read_rope_sensor():
    global Rope_S
    global buffer_dyn_Srope
    global buffer_weight_Srope
    Srope = DaYangSensor('/dev/ttyUSB0',0)
    try:
        while True:
            Rope_S = Srope.read_angles()
            buffer_dyn_Srope.append(Rope_S)
            buffer_weight_Srope.append(Rope_S)
            time.sleep(0.001)
    except KeyboardInterrupt:
        Srope.close()
        print("Ctrl-C is pressed!")

def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
    global buffer_dyn_Stouch
    global buffer_weight_Stouch
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server started, waiting for connection on {port}...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                msg = data.decode()
                Touch_S = int(msg.split()[-1])
                buffer_dyn_Stouch.append(Touch_S)
                buffer_weight_Stouch.append(Touch_S)

class ImpedanceController:
    def __init__(self):
        self.Md = 15  # 期望质量 (虚拟质量)
        self.Bd = 0.1  # 期望阻尼
        self.current_velocity = 0.0
        self.current_acceleration = 0.0
        
    def impedance_control(self, rope_force, gravity, cur_pos_abs):
        human_force = gravity - rope_force
        dead_zone = 20
        if abs(human_force) < dead_zone:
            human_force = 0
        max_acc = 35 + abs(1962000000-cur_pos_abs)/(1962000000-1953000000)*25
        max_vel = 8000
        self.current_acceleration = (human_force - self.Bd * self.current_velocity) / self.Md
        self.current_acceleration = np.clip(self.current_acceleration, -max_acc, max_acc)
        self.current_velocity += self.current_acceleration
        self.current_velocity = np.clip(self.current_velocity, -max_vel, max_vel)
        return self.current_velocity

def nonlinear_control(pres):
    p_left = 300
    y_left = 0
    p_right = 1500
    y_right = 1.5
    p_cent = (p_left + p_right) / 2  # 600
    mid_start = 500  # 中间直线段开始
    mid_end = 900    # 中间直线段结束
    if pres <= mid_start:
        # 左侧二次曲线区域
        x_norm = (pres - p_left) / (mid_start - p_left)
        a = y_left - 1.0  # 目标值改为1.0（中间直线段的值）
        b = 2 * (1.0 - y_left)
        coef = a * x_norm**2 + b * x_norm + y_left
    elif pres <= mid_end:
        coef = 1.0
    else:
        x_norm = (pres - mid_end) / (p_right - mid_end)
        a = y_right - 1.0  # 目标值改为1.0（中间直线段的值）
        b = 0
        coef = a * x_norm**2 + b * x_norm + 1.0
    coef = np.clip(coef, 0,2)
    return coef

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        imp_controller = ImpedanceController()
        threading.Thread(target=read_rope_sensor, args=(), daemon=True).start()
        threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()
        threading.Thread(target=read_cur_pos, args=(myXYZ,InitPos,), daemon=True).start()
        while Touch_S==0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        Vgoal=0
        Touch_valve=50
        Pnum=0
        mode=0

        with open('sensor_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['record_num', 'mode', 'Rope_S', 'Touch_S', 'cur_pos_abs', 'cur_vel', 'cur_acc'])

        while True:
            time.sleep(0.001)
            if np.mean(buffer_dyn_Stouch) < Touch_valve:
                mode = 3 # 松弛模式
                gravity = 250
                Vgoal = imp_controller.impedance_control(np.mean(buffer_dyn_Srope), gravity, cur_pos_abs)
                myXYZ.AxisMode_Jog(3, 30, Vgoal)
            else:
                mode = 2 # 负载模式
                gravity = 2000
                while np.mean(buffer_dyn_Stouch) >= Touch_valve:
                    time.sleep(0.001)
                    coef = nonlinear_control(np.mean(buffer_dyn_Stouch))
                    gravity=gravity+(gravity*coef-gravity)*0.02
                    gravity = np.clip(gravity, 100, 3000)
                    Vgoal = imp_controller.impedance_control(np.mean(buffer_dyn_Srope), gravity, cur_pos_abs)
                    myXYZ.AxisMode_Jog(3, 30, Vgoal)
                    Pnum += 1
                    if Pnum % 10 == 0:
                        print(
                            'Mode:', mode,
                            'Touch_F:', int(np.mean(buffer_dyn_Stouch)),
                            # 'Rope_F:', int(np.mean(buffer_dyn_Srope)),
                            # 'Human_F:', int(gravity - np.mean(buffer_dyn_Srope)),
                            # 'Vgoal:', int(Vgoal),
                            # 'diff:', int(imp_controller.current_acceleration),
                            'coef:', round(coef,2),
                            'gravity:', round(gravity,2),
                            )
                        with open('sensor_data.csv', 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([int(np.mean(buffer_dyn_Stouch)), coef, gravity])

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()