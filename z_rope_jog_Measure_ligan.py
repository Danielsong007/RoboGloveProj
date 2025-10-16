import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque
import threading

# 全局变量存储传感器数据
Rope_S = 0
Touch_S = 0

def read_rope_sensor(sensor):
    global Rope_S
    while True:
        Rope_S = sensor.read_angles()
        time.sleep(0.005)

def read_touch_sensor(sensor):
    global Touch_S
    while True:
        Touch_S = sensor.read_data()[1]
        time.sleep(0.005)
        print('in',Touch_S)


# def is_motor_rising(positions):
#     if len(positions) < 2:
#         return False
#     # 计算位置变化趋势（使用线性回归斜率）
#     x = np.arange(len(positions))
#     slope, _ = np.polyfit(x, positions, 1)
#     # 如果斜率为正且统计显著，则认为电机在上升
#     return slope > 0.1  # 调整阈值以适应您的系统

# if Touch_S>Touch_valve and Weight>0 and is_motor_rising(list(pos100_buffer)): # Update weight
    # avg_force = np.mean(list(Srope100_buffer))
    # Weight=avg_force

def main():
    try:
        myXYZ = xyz_utils()
        print('start enable')
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()

        Vgoal_L=0
        Vgoal_N=0
        Weight=0
        Touch_valve=15000
        Srope_buffer = deque(maxlen=3)  # 自动丢弃旧数据
        Stouch_buffer = deque(maxlen=3)  # 自动丢弃旧数据

        Srope = DaYangSensor('/dev/ttyUSB0',0)
        Sligan = LiganSensor(port='/dev/ttyUSB3')
        threading.Thread(target=read_rope_sensor, args=(Srope,), daemon=True).start()
        threading.Thread(target=read_touch_sensor, args=(Sligan,), daemon=True).start()

        while True:
            time.sleep(0.02)

            # Srope_buffer.append(Rope_S)
            # Ave_Rope_S=np.mean(Srope_buffer)
            # Stouch_buffer.append(Touch_S)
            # Ave_Touch_S=np.mean(Stouch_buffer)

            # # if Ave_Touch_S>Touch_valve and Weight==0 and 1964000000-(myXYZ.Get_Pos(3)+InitPos)<4000000:
            # if Ave_Touch_S>Touch_valve and Weight==0:
            #     print('Enter Measurement!!!')
            #     mode=1 # Measure Mode
            #     CurPos=myXYZ.Get_Pos(3)
            #     travel_limit=1000000
            #     cur_time = time.time()
            #     while Rope_S<500 and (CurPos-myXYZ.Get_Pos(3))<travel_limit and (time.time()-cur_time)<1:
            #         # Vgoal=2000+0.005*Touch_S
            #         Vgoal=2000
            #         myXYZ.AxisMode_Jog(3,30,Vgoal)
            #         print('Waiting Lifting, Rope_S:',Rope_S, 'Touch_S:',Touch_S, 'Ave_Touch_S:',Ave_Touch_S, 'Travel:',(CurPos-myXYZ.Get_Pos(3))/travel_limit, 'Elapsed time:',time.time()-cur_time)
            #         time.sleep(0.01)
            #     if Rope_S>500:
            #         samples_ropeS=[]
            #         samples_TouchS=[]
            #         for i in range(1,50):
            #             samples_ropeS.append(Rope_S)
            #             samples_TouchS.append(Touch_S)
            #             time.sleep(0.01)
            #         Weight=np.mean(samples_ropeS)+0.02*np.mean(samples_TouchS)
            #         print('Measured Success! Weight:',Weight)
            #     else:
            #         print('Measured Failure: Over Travel or Time!')
            #         Vgoal=0
            #         myXYZ.AxisMode_Jog(3,30,Vgoal)
            #         time.sleep(1.5)
            #     Vgoal_L=Vgoal

            # if Ave_Touch_S<Touch_valve:
            #     Weight=0
            #     mode=3 # Lossen Mode
            #     err=40-Ave_Rope_S
            #     Vgoal_N=40*err
            # else:
            #     mode=2 # Load Mode
            #     err=Weight-Ave_Rope_S
            #     if abs(err)<100:
            #         err=0
            #     Vgoal_N=15*err

            # diff=(Vgoal_N-Vgoal_L)*0.01
            # Max_diff=30
            # diff = np.clip(diff, -Max_diff, Max_diff)
            # Max_Vel=4000
            # Vgoal = np.clip(Vgoal_L+diff, -Max_Vel, Max_Vel)
            # myXYZ.AxisMode_Jog(3,30,Vgoal)
            # Vgoal_L=Vgoal
            # print('Mode:',mode, 'Rope_S:',Rope_S, 'Ave_Touch_S:',Ave_Touch_S, 'Vgoal',int(Vgoal), 'Diff:',int(diff), 'Weight', int(Weight))

            # myXYZ.AxisMode_Jog(3,30,-2000)


    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        # Srope.close()
        # Sligan.close()
        sys.exit(0)

if __name__ == "__main__":
    main()



