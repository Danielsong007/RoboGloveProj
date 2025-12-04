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
import matplotlib.pyplot as plt


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
current_rope_force = np.mean(buffer_dyn_Srope)
current_touch_force = np.mean(buffer_dyn_Stouch)

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
    global current_rope_force
    Srope = DaYangSensor('/dev/ttyUSB0',0)
    try:
        while True:
            Rope_S = Srope.read_angles()
            buffer_dyn_Srope.append(Rope_S)
            buffer_weight_Srope.append(Rope_S)
            current_rope_force = np.mean(buffer_dyn_Srope)
            time.sleep(0.001)
    except KeyboardInterrupt:
        Srope.close()
        print("Ctrl-C is pressed!")

def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
    global buffer_dyn_Stouch
    global buffer_weight_Stouch
    global current_touch_force
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
                current_touch_force = np.mean(buffer_dyn_Stouch)

class IC_ROPE:
    def __init__(self):
        self.Md = 2  # 期望质量 (虚拟质量)
        self.Bd = 0.1  # 期望阻尼
        self.current_velocity = 0.0
        self.current_acceleration = 0.0
        
    def impedance_control(self, rope_force, balance_force, cur_pos_abs):
        human_force = balance_force - rope_force
        dead_zone = 10
        if abs(human_force) < dead_zone:
            human_force = 0
        max_acc = 40
        max_vel = 4000
        self.current_acceleration = (human_force - self.Bd * self.current_velocity) / self.Md
        self.current_acceleration = np.clip(self.current_acceleration, -max_acc, max_acc)
        self.current_velocity += self.current_acceleration
        self.current_velocity = np.clip(self.current_velocity, -max_vel, max_vel)
        return self.current_velocity

class IC_TOUCH:
    def __init__(self):
        self.Md = 5  # 期望质量 (虚拟质量)
        self.Bd = 0.1  # 期望阻尼
        self.current_velocity = 4000
        self.current_acceleration = 0.0
        
    def impedance_control(self, touch_force, balance_force, cur_pos_abs):
        human_force = touch_force - balance_force
        dead_zone = 10
        if abs(human_force) < dead_zone:
            human_force = 0
        max_acc = 35 + abs(1962000000-cur_pos_abs)/(1962000000-1953000000)*25
        max_vel = 5000
        self.current_acceleration = (human_force - self.Bd * self.current_velocity) / self.Md
        self.current_acceleration = np.clip(self.current_acceleration, -max_acc, max_acc)
        self.current_velocity += self.current_acceleration
        self.current_velocity = np.clip(self.current_velocity, -max_vel, max_vel)
        return self.current_velocity

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        threading.Thread(target=read_rope_sensor, args=(), daemon=True).start()
        threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()
        threading.Thread(target=read_cur_pos, args=(myXYZ,InitPos,), daemon=True).start()
        while Touch_S==0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        Vgoal=0
        Touch_valve=100
        Pnum=0
        mode=0
        ic_rope = IC_ROPE()
        ic_touch = IC_TOUCH()

        plt_touch=[]
        plt_human=[]
        plt_acc=[]
        plt_vgoal=[]
        while True:
            time.sleep(0.001)           
            if current_touch_force < Touch_valve:
                mode = 3 # 松弛模式
                balance_force = 150
                Vgoal = ic_rope.impedance_control(current_rope_force, balance_force, cur_pos_abs)
                if Pnum % 10 == 0:
                    print(
                        'Mode:', mode,
                        'Touch_F:', int(current_touch_force),
                        'Rope_F:', int(current_rope_force),
                        'Human_F:', int(balance_force - current_rope_force),
                        'Vgoal:', int(Vgoal),
                        'diff:', int(ic_rope.current_acceleration),
                        #   'dt:', round(dt,5),
                        )
            else:
                mode = 2 # 负载模式
                balance_force = 900
                Vgoal = ic_touch.impedance_control(current_touch_force, balance_force, cur_pos_abs)
                if Pnum % 10 == 0:
                    print(
                        'Mode:', mode,
                        'Touch_F:', int(current_touch_force),
                        'Rope_F:', int(current_rope_force),
                        'Human_F:', int(current_touch_force-balance_force),
                        'Vgoal:', int(Vgoal),
                        'diff:', int(ic_touch.current_acceleration),
                        #   'dt:', round(dt,5),
                        )
                plt_touch.append(int(current_touch_force))
                plt_human.append(int(current_touch_force-balance_force)/10)
                plt_acc.append(int(ic_touch.current_acceleration))
                plt_vgoal.append(int(Vgoal)/100)
            myXYZ.AxisMode_Jog(3, 30, Vgoal)
            Pnum += 1

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        
        x = range(len(plt_human))
        # plt.plot(x, plt_touch, 'b-', label='touch', linewidth=1.5)
        plt.plot(x, plt_human, 'r-', label='human/10', linewidth=1.5)
        plt.plot(x, plt_acc, 'g-', label='acc', linewidth=1.5)
        plt.plot(x, plt_vgoal, 'k-', label='vgoal/100', linewidth=1.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        sys.exit(0)

if __name__ == "__main__":
    main()