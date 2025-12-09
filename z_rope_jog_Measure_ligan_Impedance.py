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

def read_cur_pos(myXYZ, InitPos):
    global cur_pos_abs
    global cur_vel
    global cur_acc
    global buffer_rising_CurPos
    global rising_slope
    last_time = time.time()
    last_pos = myXYZ.Get_Pos(3)+InitPos
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
        
    def impedance_control(self, rope_force, Weight, cur_pos_abs):
        human_force = Weight - rope_force
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
        Touch_valve=100
        Pnum=0
        mode=0
        Weight=0

        with open('sensor_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['record_num', 'mode', 'Rope_S', 'Touch_S', 'cur_pos_abs', 'cur_vel', 'cur_acc'])
        record_num=0
        record_step=1
        while True:
            if record_num % record_step == 0:
                with open('sensor_data.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([int(record_num/record_step), mode, Rope_S, Touch_S, cur_pos_abs, cur_vel, cur_acc])
            record_num +=1
            current_rope_force = np.mean(buffer_dyn_Srope)
            current_touch_force = np.mean(buffer_dyn_Stouch)
            
            if current_touch_force>Touch_valve and Weight<=300:
                print('Enter Measurement!!!')
                mode=1 # Measure Mode
                cur_time = time.time()
                while Rope_S<500 and (time.time()-cur_time)<1:
                    Vgoal=2000
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    print('Waiting Lifting',
                          'Rope_S:', Rope_S,
                          'Elapsed time:', int(100*(time.time()-cur_time)),
                          'ave_dyn_Srope:', int(np.mean(buffer_dyn_Srope)),
                          'ave_dyn_Stouch:', int(np.mean(buffer_dyn_Stouch)),
                          )
                    time.sleep(0.01)
                time.sleep(0.5)
                if Rope_S>500:
                    Weight = (1*np.mean(buffer_weight_Srope) + 0.5*np.mean(buffer_weight_Stouch))*1.5
                    print('Measured Success! Weight:', Weight)
                else:
                    Vgoal=0
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    time.sleep(0.5)
                    print('Measured Failure: Over Time!')
            elif current_touch_force <= Touch_valve:
                mode = 3 # 松弛模式
                Weight = 250
            else:
                mode = 2 # 负载模式
                Weight = 1700
            Vgoal = imp_controller.impedance_control(current_rope_force, Weight, cur_pos_abs)
            myXYZ.AxisMode_Jog(3, 30, Vgoal)
            Pnum += 1
            if Pnum % 10 == 0:
                print(
                      'Mode:', mode,
                      'Touch_F:', int(current_touch_force),
                      'Rope_F:', int(current_rope_force),
                      'Human_F:', int(Weight - current_rope_force),
                      'Touch_F:', int(current_touch_force),
                      'Vgoal:', int(Vgoal),
                      'diff:', int(imp_controller.current_acceleration),
                    #   '1/dt:', round(1/dt,5),
                      )
            

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()