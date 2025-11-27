import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading
import socket
import math

# 全局变量存储传感器数据
Rope_S = 0
Touch_S = 0

def read_rope_sensor():
    global Rope_S
    Srope = DaYangSensor('/dev/ttyUSB0',0)
    try:
        while True:
            Rope_S = Srope.read_angles()
            time.sleep(0.001)
    except KeyboardInterrupt:
        Srope.close()
        print("Ctrl-C is pressed!")

def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
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

threading.Thread(target=read_rope_sensor, args=(), daemon=True).start()
threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()



def nonlinear_control(pressure):
    center_pressure = 400  # 中心压力值
    min_pressure = 100     # 最小压力
    max_pressure = 900     # 最大压力
    
    if pressure < center_pressure:
        ratio = (center_pressure - pressure) / (center_pressure - min_pressure)
        down_torq = ratio ** 1.5 * 200
        up_torq = 0
    else:
        ratio = (pressure - center_pressure) / (max_pressure - center_pressure)
        up_torq = ratio ** 1.5 * 1500
        down_torq = 0
    
    # 最终速度目标
    Tgoal_N = up_torq - down_torq
    
    return Tgoal_N

def main():
    try:
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        myXYZ.AxisMode_Torque(3)
        mode=0

        while Touch_S == 0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')

        Tgoal_N = 0
        Tgoal = 0
        Srope_buffer = deque(maxlen=3)
        Stouch_buffer = deque(maxlen=3)
        
        while True:
            dt=0.001
            time.sleep(dt)
            Srope_buffer.append(Rope_S)
            Ave_Rope_S = np.mean(Srope_buffer)
            Stouch_buffer.append(Touch_S)
            Ave_Touch_S = np.mean(Stouch_buffer)

            Tgoal_N = nonlinear_control(Ave_Touch_S)

            diff=(Tgoal_N-Tgoal)*0.9
            Max_Torq=2000
            Tgoal = np.clip(Tgoal+diff, -Max_Torq, Max_Torq)
            Tgoal = int(Tgoal)

            CurPos=myXYZ.Get_Pos(3)
            if CurPos>1962000000:
                Tgoal=50
            if CurPos<1953000000:
                Tgoal=-50
            myXYZ.Set_Torque(3,-Tgoal)
            myXYZ.Set_Torque(3,-Tgoal)
            T_Sent,T_Actual,V_actual=myXYZ.Read_Paras(3)
            print('T_Goal:',Tgoal, 'T_Sent:',-T_Sent, 'T_Actual: ',-T_Actual, 'diff:',int(diff))

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        myXYZ.Set_Torque(3,0)
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()


