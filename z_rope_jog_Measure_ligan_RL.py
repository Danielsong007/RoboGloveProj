import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading
import socket


def read_rope_sensor():
    global Rope_S
    global Ave_Rope_S
    Srope_buffer = deque(maxlen=3)  # 自动丢弃旧数据
    Srope = DaYangSensor('/dev/ttyUSB0',0)
    try:
        while True:
            Rope_S = Srope.read_angles()
            Srope_buffer.append(Rope_S)
            Ave_Rope_S=np.mean(Srope_buffer)
            time.sleep(0.001)
    except KeyboardInterrupt:
        Srope.close()
        print("Ctrl-C is pressed!")

def touch_sensor_server(host='0.0.0.0', port=65432):
    global Touch_S
    global Ave_Touch_S
    Stouch_buffer = deque(maxlen=3)  # 自动丢弃旧数据
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
                Stouch_buffer.append(Touch_S)
                Ave_Touch_S=np.mean(Stouch_buffer)

Rope_S = 0
Touch_S = 0
Ave_Rope_S = 0
Ave_Touch_S = 0
threading.Thread(target=read_rope_sensor, args=(), daemon=True).start()
threading.Thread(target=touch_sensor_server, args=(), daemon=True).start()

# def is_motor_rising(positions):
#     if len(positions) < 2:
#         return False
#     # 计算位置变化趋势（使用线性回归斜率）
#     x = np.arange(len(positions))
#     slope, _ = np.polyfit(x, positions, 1)
#     # 如果斜率为正且统计显著，则认为电机在上升
#     return slope > 0.1  # 调整阈值以适应您的系统

# if Touch_S>Touch_valve and Weight>0 and is_motor_rising(list(pos100_buffer)): # Update weight
#     avg_force = np.mean(list(Srope100_buffer))
#     Weight=avg_force

def main():
    try:
        myXYZ = xyz_utils()
        print('Start Enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        while Touch_S==0:
            time.sleep(0.1)
            print('Waiting Touch Data!!')
        # time.sleep(10)
        
        Vgoal=0
        Vgoal_N=0
        Weight=0
        Touch_valve=200
        Pnum=0

        while True:
            time.sleep(0.001)
            # if Ave_Touch_S>Touch_valve and Weight==0 and 1964000000-(myXYZ.Get_Pos(3)+InitPos)<4000000:
            if Ave_Touch_S>Touch_valve and Weight==0:
                print('Enter Measurement!!!')
                mode=1 # Measure Mode
                CurPos=myXYZ.Get_Pos(3)
                cur_time = time.time()
                while Rope_S<500 and (time.time()-cur_time)<1:
                    # Vgoal=2000+0.005*Touch_S
                    Vgoal=2000
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    print('Waiting Lifting, Rope_S:',Rope_S, 'Touch_S:',Touch_S, 'Ave_Touch_S:',Ave_Touch_S, 'Elapsed time:',time.time()-cur_time)
                    time.sleep(0.01)
                if Rope_S>500:
                    samples_ropeS=[]
                    samples_TouchS=[]
                    for i in range(1,50):
                        samples_ropeS.append(Rope_S)
                        samples_TouchS.append(Touch_S)
                        time.sleep(0.01)
                    Weight=np.mean(samples_ropeS)+0.2*np.mean(samples_TouchS)
                    print('Measured Success! Weight:', Weight)
                else:
                    Vgoal=0
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    time.sleep(1.5)
                    print('Measured Failure: Over Time!')
            if Ave_Touch_S<Touch_valve:
                Weight=0
                mode=3 # Lossen Mode
                err=40-Ave_Rope_S
                Vgoal_N=40*err
            else:
                mode=2 # Load Mode
                err=Weight-Ave_Rope_S
                if abs(err)<150:
                    err=0
                Vgoal_N=15*err

            diff=(Vgoal_N-Vgoal)*0.01
            Max_diff=15
            diff = np.clip(diff, -Max_diff, Max_diff)
            Max_Vel=4000
            Vgoal = np.clip(Vgoal+diff, -Max_Vel, Max_Vel)
            myXYZ.AxisMode_Jog(3,30,Vgoal)

            Pnum += 1
            if Pnum % 29 == 0:
                print('Mode:',mode, 'Ave_Rope_S:',int(Ave_Rope_S), 'Ave_Touch_S:',int(Ave_Touch_S), 'Vgoal_N',int(Vgoal_N), 'Vgoal',int(Vgoal), 'diff:',int(diff), 'Weight', int(Weight))

            # myXYZ.AxisMode_Jog(3,30,-2000)

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        sys.exit(0)

if __name__ == "__main__":
    main()



