import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
from sensors.ligan import LiganSensor
import numpy as np
from collections import deque

def is_motor_rising(positions):
    if len(positions) < 2:
        return False
    # 计算位置变化趋势（使用线性回归斜率）
    x = np.arange(len(positions))
    slope, _ = np.polyfit(x, positions, 1)
    # 如果斜率为正且统计显著，则认为电机在上升
    return slope > 0.1  # 调整阈值以适应您的系统

def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0',0)
        Sligan=LiganSensor(port='/dev/ttyUSB3')
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog(3)

        Vgoal_L=0
        Vgoal_N=0
        Srope_buffer = deque(maxlen=3)  # 自动丢弃旧数据
        pos100_buffer = deque(maxlen=100)  # 电机位置
        Srope100_buffer = deque(maxlen=100)    # 拉力传感器数值

        while True:
            time.sleep(0.01)
            Rope_S=Srope.read_angles()
            Touch_S=Sligan.read_data()[1]

            Srope_buffer.append(Rope_S)
            Ave_Rope_S=np.mean(Srope_buffer)
            pos100_buffer.append(myXYZ.Get_Pos(3))
            Srope100_buffer.append(Rope_S)
            
            if Touch_S>50: # Update weight
                # 检查是否持续上升
                if is_motor_rising(list(pos100_buffer)):
                    # 计算拉力传感器平均值
                    avg_force = np.mean(list(Srope100_buffer))
                Weight=avg_force
            else:
                if Touch_S>50:
                    mode=2 # Loop Mode
                    err=Weight-Ave_Rope_S
                    err_min=100
                    if err>-err_min and err<err_min:
                        err=0
                    Vgoal_N=15*err
                else:
                    mode=3 # Lossen Mode
                    err=40-Ave_Rope_S
                    Vgoal_N=40*err
                    Flag_Weight=0
                
                diff=(Vgoal_N-Vgoal_L)*0.03
                Max_diff=150
                if diff>Max_diff:
                    diff=Max_diff
                if diff<-Max_diff:
                    diff=-Max_diff
                Vgoal=Vgoal_L+diff

                Max_Vel=4000
                Min_Vel=-4000
                if Vgoal<Min_Vel:
                    Vgoal=Min_Vel
                if Vgoal>Max_Vel:
                    Vgoal=Max_Vel
                myXYZ.AxisMode_Jog(3,30,Vgoal)
                Vgoal_L=Vgoal

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


        
