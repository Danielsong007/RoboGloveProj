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

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        while Touch_S==0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        
        Vgoal_N=0
        Vgoal=0
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

            if Ave_Touch_S<Touch_valve:
                mode=3 # Lossen Mode
                err=50-Ave_Rope_S
                Vgoal_N=40*err
            else:
                mode=2 # Load Mode
                Vgoal_N=Ave_Touch_S*0.3-Ave_Rope_S*3

            diff=(Vgoal_N-Vgoal)*0.01
            Max_Vel=4000
            Vgoal = np.clip(Vgoal+diff, -Max_Vel, Max_Vel)
            myXYZ.AxisMode_Jog(3,30,int(Vgoal))

            Pnum += 1
            if Pnum % 29 == 0:
                print('Mode:',mode, 'Ave_Rope_S:',int(Ave_Rope_S), 'Ave_Touch_S:',int(Ave_Touch_S), 'Vgoal_N',int(Vgoal_N), 'Vgoal',int(Vgoal), 'diff:',int(diff), 'Weight', int(Weight))

            # myXYZ.AxisMode_Jog(3,30,-2000)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()



