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
Pres_S = 0
Touch_S = 0
cur_pos_abs = 0
cur_vel = 0
cur_acc = 0
Vgoal=0

buffer_dyn_Srope = deque([0.0]*3, maxlen=3)
buffer_dyn_Spres = deque([0.0]*3, maxlen=3)
buffer_dyn_Stouch = deque([0.0]*3, maxlen=3)
current_rope_force = np.mean(buffer_dyn_Srope)
current_pres_force = np.mean(buffer_dyn_Spres)
current_touch_force = np.mean(buffer_dyn_Stouch)
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
    global current_rope_force
    Srope = DaYangSensor('/dev/ttyUSB0',0)
    try:
        while True:
            Rope_S = Srope.read_angles()
            buffer_dyn_Srope.append(Rope_S)
            current_rope_force = np.mean(buffer_dyn_Srope)
    except KeyboardInterrupt:
        Srope.close()
        print("Ctrl-C is pressed!")

def read_pres_sensor():
    global Pres_S
    global buffer_dyn_Spres
    global current_pres_force
    Spres = DaYangSensor('/dev/ttyUSB1',1)
    try:
        while True:
            Pres_S = Spres.read_angles()
            buffer_dyn_Spres.append(Pres_S)
            current_pres_force = np.mean(buffer_dyn_Spres)
    except KeyboardInterrupt:
        Spres.close()
        print("Ctrl-C is pressed!")

def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
    global buffer_dyn_Stouch
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
                current_touch_force = np.mean(buffer_dyn_Stouch)

class IC_ROPE:
    def __init__(self):
        self.Md = 1  # 期望质量 (虚拟质量)
        self.Bd = 0.1  # 期望阻尼
        self.current_velocity = 0.0
        self.current_acceleration = 0.0
        
    def impedance_control(self, rope_force, balance_force, cur_pos_abs):
        human_force = balance_force - rope_force
        dead_zone = 10
        if abs(human_force) < dead_zone:
            human_force = 0
        # max_acc = 40
        max_vel = 8000
        self.current_acceleration = (human_force - self.Bd * self.current_velocity) / self.Md
        # self.current_acceleration = np.clip(self.current_acceleration, -max_acc, max_acc)
        self.current_velocity += self.current_acceleration
        self.current_velocity = np.clip(self.current_velocity, -max_vel, max_vel)
        return self.current_velocity

class IC_TOUCH:
    def __init__(self):
        self.Md = 20  # 期望质量 (虚拟质量)
        self.Bd = 0.7  # 期望阻尼
        self.current_acceleration = 0.0
        
    def impedance_control(self, human_force, cur_pos_abs, Vgoal):
        self.current_velocity = Vgoal
        dead_zone = 10
        if abs(human_force) < dead_zone:
            human_force = 0
        # max_acc = 300
        self.current_acceleration = (human_force - self.Bd * self.current_velocity) / self.Md
        # self.current_acceleration = np.clip(self.current_acceleration, -max_acc, max_acc)
        self.current_velocity += self.current_acceleration
        self.current_velocity = np.clip(self.current_velocity, -5000, 8000)
        return self.current_velocity

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        threading.Thread(target=read_rope_sensor, args=(), daemon=True).start()
        threading.Thread(target=read_pres_sensor, args=(), daemon=True).start()
        threading.Thread(target=read_cur_pos, args=(myXYZ,InitPos,), daemon=True).start()

        # threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()
        # while Touch_S==0:
        #     time.sleep(0.1)
        #     print('Waiting Touch Data!!')
        
        Pres_valve=30
        Pnum=0
        mode=0
        last_mode=0
        ic_rope = IC_ROPE()
        ic_touch = IC_TOUCH()

        plt_pres=[]
        plt_human=[]
        plt_acc=[]
        plt_vgoal=[]
        plt_BdN=[]

        while True:
            time.sleep(0.001)           
            if current_pres_force < Pres_valve:
                mode = 3 # 松弛模式
                last_mode = mode
                balance_force = 150
                Vgoal = ic_rope.impedance_control(current_rope_force, balance_force, cur_pos_abs)
                if Pnum % 10 == 0:
                    print(
                        'Mode:', mode,
                        'Pres_F:', int(current_pres_force),
                        'Rope_F:', int(current_rope_force),
                        'Human_F:', int(balance_force - current_rope_force),
                        'Vgoal:', int(Vgoal),
                        'diff:', int(ic_rope.current_acceleration),
                        #   'dt:', round(dt,5),
                        )
            else: # 位置可用于mode3切换判断_空中位置区间松手也不该切为MODE3
                if last_mode == 3:
                    print('Init enter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    Vgoal = 4000
                    myXYZ.AxisMode_Jog(3, 30, Vgoal)
                    time.sleep(0.3)
                mode = 2 # 负载模式
                last_mode = mode
                balance_force = 1000
                human_force = current_pres_force - balance_force
                if human_force < 0:
                    human_force = human_force*6
                Vgoal = ic_touch.impedance_control(human_force, cur_pos_abs, Vgoal)
                if Pnum % 10 == 0:
                    print(
                        'Mode:', mode,
                        'Pres_F:', int(current_pres_force),
                        # 'Rope_F:', int(current_rope_force),
                        'Human_F:', int(human_force),
                        'Vgoal:', int(Vgoal),
                        'diff:', int(ic_touch.current_acceleration),
                        #   'dt:', round(dt,5),
                        )
                plt_pres.append(int(current_pres_force))
                plt_human.append(int(human_force)/10)
                plt_acc.append(int(ic_touch.current_acceleration))
                plt_vgoal.append(int(Vgoal)/10)
                plt_BdN.append(int(ic_touch.Bd*Vgoal)/10)
            myXYZ.AxisMode_Jog(3, 30, Vgoal)
            Pnum += 1

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
    finally:
        myXYZ.SafeQuit()
        
        x = range(len(plt_human))
        plt.plot(x, plt_BdN, 'b-', label='BdN/10', linewidth=1.5)
        plt.plot(x, plt_human, 'r-', label='human/10', linewidth=1.5)
        plt.plot(x, plt_acc, 'g-', label='acc', linewidth=1.5)
        plt.plot(x, plt_vgoal, 'k-', label='vgoal/100', linewidth=1.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        sys.exit(0)

if __name__ == "__main__":
    main()