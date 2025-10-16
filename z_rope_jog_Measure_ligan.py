import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque

# def is_motor_rising(positions):
#     if len(positions) < 2:
#         return False
#     # 计算位置变化趋势（使用线性回归斜率）
#     x = np.arange(len(positions))
#     slope, _ = np.polyfit(x, positions, 1)
#     # 如果斜率为正且统计显著，则认为电机在上升
#     return slope > 0.1  # 调整阈值以适应您的系统

# if Touch_S>Touch_valve and flag_weight==1 and is_motor_rising(list(pos100_buffer)): # Update weight
    # avg_force = np.mean(list(Srope100_buffer))
    # Weight=avg_force

def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0',0)
        Sligan=LiganSensor(port='/dev/ttyUSB3')
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()

        Vgoal_L=0
        Vgoal_N=0
        flag_weight=0
        Touch_valve=80
        Srope_buffer = deque(maxlen=3)  # 自动丢弃旧数据

        while True:
            time.sleep(0.01)
            Rope_S=Srope.read_angles()
            Touch_S=Sligan.read_data()[1]

            Srope_buffer.append(Rope_S)
            Ave_Rope_S=np.mean(Srope_buffer)
            
            if Touch_S>Touch_valve and flag_weight==0 and 1964000000-(myXYZ.Get_Pos(3)+InitPos)<4000000:
                print('Initial Measurement!!!')
                mode=1 # Measure Mode
                CurPos=myXYZ.Get_Pos(3)
                travel_limit=1000000
                while Rope_S<500 and (CurPos-myXYZ.Get_Pos(3))<travel_limit:
                    Touch_S=Sligan.read_data()[1]
                    Vgoal=2000+0.1*Touch_S
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    Rope_S=Srope.read_angles()
                    print('Waiting Lifting, Rope_S:',Rope_S,'Travel:',(CurPos-myXYZ.Get_Pos(3))/travel_limit)
                    time.sleep(0.01)
                if Rope_S>500:
                    print('Measured Success! Weight:',Weight)
                    flag_weight=1
                    samples_ropeS=[]
                    samples_TouchS=[]
                    for i in range(1,50):
                        Rope_S=Srope.read_angles()
                        samples_ropeS.append(Rope_S)
                        Touch_S=Sligan.read_data()[1]
                        samples_TouchS.append(Touch_S)
                        time.sleep(0.01)
                    Weight=np.mean(samples_ropeS)+0.2*np.mean(samples_TouchS)
                else:
                    print('Measured Failure: Over Travel')
                    flag_weight=0
                    Vgoal=0
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    time.sleep(1.5)
                Vgoal_L=Vgoal

            if Touch_S<Touch_valve:
                print('Loosen mode')
                mode=3 # Lossen Mode
                flag_weight=0
                err=40-Ave_Rope_S
                Vgoal_N=40*err
            else:
                print('Load mode')
                mode=2 # Load Mode
                flag_weight=1
                err=Weight-Ave_Rope_S
                if abs(err)<100:
                    err=0
                Vgoal_N=15*err

            diff=(Vgoal_N-Vgoal_L)*0.03
            Max_diff=150
            diff = np.clip(diff, -Max_diff, Max_diff)
            Max_Vel=4000
            Vgoal = np.clip(Vgoal_L+diff, -Max_Vel, Max_Vel)
            myXYZ.AxisMode_Jog(3,30,Vgoal)
            Vgoal_L=Vgoal

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Sligan.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


        
