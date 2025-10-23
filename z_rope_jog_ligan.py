import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading
import socket

# 全局变量存储传感器数据
Rope_S = 0
Touch_S = 0

def read_rope_sensor(sensor):
    global Rope_S
    try:
        while True:
            Rope_S = sensor.read_angles()
            time.sleep(0.005)
    except KeyboardInterrupt:
        sensor.close()
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
                Touch_S = int(msg.split()[-1])  # 分割后取最后一个元素
                # print('Received',Touch_S)

Srope = DaYangSensor('/dev/ttyUSB0',0)
threading.Thread(target=read_rope_sensor, args=(Srope,), daemon=True).start()
threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        while Touch_S==0:
            time.sleep(0.05)
            print('Waiting Touch Data!!')
        
        Vgoal_L=0
        Vgoal_N=0
        Weight=0
        Touch_valve=2000
        Srope_buffer = deque(maxlen=3)  # 自动丢弃旧数据
        Stouch_buffer = deque(maxlen=3)  # 自动丢弃旧数据
        Pnum=0

        while True:
            time.sleep(0.001)
            Srope_buffer.append(Rope_S)
            Ave_Rope_S=np.mean(Srope_buffer)
            Stouch_buffer.append(Touch_S)
            Ave_Touch_S=np.mean(Stouch_buffer)

            # if Ave_Touch_S<Touch_valve:
            #     mode=3 # Lossen Mode
            #     err=50-Ave_Rope_S
            #     Vgoal_N=40*err
            # else:
            #     # time.sleep(0.5)
            #     mode=2 # Load Mode
            #     Vgoal_N=Ave_Touch_S*0.3-Ave_Rope_S*1.5
            
            mode=2 # Load Mode
            Vgoal_N=Ave_Touch_S*0.3-Ave_Rope_S*5

            diff=(Vgoal_N-Vgoal_L)*10
            Max_diff=20
            diff = np.clip(diff, -Max_diff, Max_diff)
            Max_Vel=4000
            Vgoal = np.clip(Vgoal_L+diff, -Max_Vel, Max_Vel)
            # print(Vgoal_L,diff,Vgoal)
            myXYZ.AxisMode_Jog(3,30,int(Vgoal))
            Vgoal_L=Vgoal

            Pnum=Pnum+1
            if Pnum==29:
                print('Mode:',mode, 'Ave_Rope_S:',int(Ave_Rope_S), 'Ave_Touch_S:',int(Ave_Touch_S), 'Vgoal_N',int(Vgoal_N), 'Vgoal',int(Vgoal), 'Diff:',int(diff))
                Pnum=0

            # myXYZ.AxisMode_Jog(3,30,-2000)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()



