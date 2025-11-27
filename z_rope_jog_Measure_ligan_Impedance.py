import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading
import socket

Rope_S = 0
Touch_S = 0
cur_pos_abs = 0

buffer_dyn_Srope = deque([0.0]*3, maxlen=3)
buffer_weight_Srope = deque([0.0]*30, maxlen=50)
buffer_dyn_Stouch = deque([0.0]*3, maxlen=3)
buffer_weight_Stouch = deque([0.0]*30, maxlen=50)
buffer_rising_CurPos = deque([0]*5, maxlen=5)
rising_slope = 0

def read_cur_pos(myXYZ, InitPos):
    global cur_pos_abs
    global buffer_rising_CurPos
    global rising_slope
    while True:
        cur_pos_abs = myXYZ.Get_Pos(3)+InitPos
        buffer_rising_CurPos.append(cur_pos_abs)
        x = np.arange(len(buffer_rising_CurPos))*1000
        rising_slope, _ = np.polyfit(x, -np.array(buffer_rising_CurPos), 1)
        time.sleep(0.001)

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
        
    def impedance_control(self, rope_force, gravity_force, cur_pos_abs):
        human_force = gravity_force - rope_force
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
        last_time = time.time()
        
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            time.sleep(0.001)
            current_rope_force = np.mean(buffer_dyn_Srope)
            current_touch_force = np.mean(buffer_dyn_Stouch)
            
            if current_touch_force < Touch_valve:
                mode = 3 # 松弛模式
                gravity_force = 250
            else:
                mode = 2 # 负载模式
                gravity_force = 1700
            Vgoal = imp_controller.impedance_control(current_rope_force, gravity_force, cur_pos_abs)
            myXYZ.AxisMode_Jog(3, 30, Vgoal)
            Pnum += 1
            if Pnum % 10 == 0:
                print(
                      'Mode:', mode,
                      'Touch_F:', int(current_touch_force),
                      'Rope_F:', int(current_rope_force),
                      'Human_F:', int(gravity_force - current_rope_force),
                      'Touch_F:', int(current_touch_force),
                      'Vgoal:', int(Vgoal),
                      'diff:', int(imp_controller.current_acceleration),
                    #   'dt:', round(dt,5),
                      )

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()