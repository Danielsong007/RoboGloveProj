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
import matplotlib.pyplot as plt


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

# def nonlinear_control(pres, tension):
#     threshold_pres = 400
#     if pres < threshold_pres:
#         up_speed = (pres / threshold_pres) * 500
#     else:
#         ratio = (pres - threshold_pres) / threshold_pres
#         up_speed = 500 + ratio ** 0.8 * 2500
    
#     # 拉力控制：向下速度分量
#     threshold_tens = 1000
#     if tension < threshold_tens:
#         down_speed = (tension / threshold_tens) * 500
#     else:
#         # 大于1000：指数快速增加
#         ratio = (tension - threshold_tens) / threshold_tens
#         down_speed = 500 + ratio ** 0.8 * 2500
    
#     # 最终速度目标
#     Vgoal_N = up_speed - down_speed
    
#     return Vgoal_N

def nonlinear_control(pres):
    if pres > 300:
        pres=(pres - 400) / 300 * 10
        Vgoal_N = 1 / (1+np.exp(-pres)) * 11000 - 384
    else:
        pres=(pres - 200) / 300 * 10
        Vgoal_N = 1 / (1+np.exp(-pres)) * 5000 - 5000 + 178
    return Vgoal_N


def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos = myXYZ.Safe_Jog()
        
        while Touch_S == 0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        
        Vgoal_N = 0
        Vgoal = 0
        Vgoal_L = 0
        Srope_buffer = deque(maxlen=3)
        Stouch_buffer = deque(maxlen=3)
        Pnum = 0

        touch_set=[]
        vgoal_set=[]

        while True:
            time.sleep(0.001)
            Srope_buffer.append(Rope_S)
            Ave_Rope_S = np.mean(Srope_buffer)
            Stouch_buffer.append(Touch_S)
            Ave_Touch_S = np.mean(Stouch_buffer)

            if Ave_Touch_S < 70:
                mode = 3  # Lossen Mode
                err = 100 - Ave_Rope_S
                Vgoal_N = 20 * err
                Vgoal_N = np.clip(Vgoal_N, -10000, 900)
            else:
                mode = 2  # Load Mode
                Vgoal_N = nonlinear_control(Ave_Touch_S)
            
            diff=(Vgoal_N-Vgoal)*0.2
            diff = np.clip(diff, -300, 400)
            Vgoal = np.clip(Vgoal+diff, -5500, 9000)
            myXYZ.AxisMode_Jog(3,30,Vgoal)

            Pnum += 1
            if Pnum % 5 == 0 and mode == 2:
                print(
                    # 'Mode:',mode,
                    # 'Rope:',int(Ave_Rope_S),
                      'Touch:',int(Ave_Touch_S),
                      'Vgoal_N',int(Vgoal_N),
                      'Vgoal',int(Vgoal),
                      'diff:',int(Vgoal - Vgoal_L),
                      )
            Vgoal_L = Vgoal


    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()