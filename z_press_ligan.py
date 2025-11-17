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

buffer_dyn_Srope = deque([0.0]*3, maxlen=3)
buffer_weight_Srope = deque([0.0]*30, maxlen=50)
buffer_dyn_Stouch = deque([0.0]*3, maxlen=3)
buffer_weight_Stouch = deque([0.0]*30, maxlen=50)
buffer_rising_CurPos = deque([0]*5, maxlen=5)
rising_slope = 0


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

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()
        while Touch_S==0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        Vgoal=0
        Vgoal_N=0
        Pnum=0
        
        while True:
            time.sleep(0.001)
            Pnum += 1
            ave_touch=np.mean(buffer_dyn_Stouch)

            if ave_touch > 300: # Up
                Vgoal_N = ave_touch*5
            elif np.mean(buffer_dyn_Stouch) > 150: # Suspend
                Vgoal_N = 0
            else: # Down
                Vgoal_N = -20*(150-ave_touch)

            diff=(Vgoal_N-Vgoal)*0.1
            Max_diff=50
            diff = np.clip(diff, -Max_diff, Max_diff)
            Max_Vel=4000
            Vgoal = np.clip(Vgoal+diff, -Max_Vel, Max_Vel)
            myXYZ.AxisMode_Jog(3,30,Vgoal)

            if Pnum % 30 == 0:
                print('ave_touch: ', ave_touch,
                      'Vgoal_N: ', Vgoal_N,
                      'Vgoal: ', int(Vgoal),
                      'diff: ', int(diff),
                      )

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()


            