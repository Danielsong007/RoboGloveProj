import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.dayang import DaYangSensor
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0',0)
        Shand = DaYangSensor('/dev/ttyUSB1',1)
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        myXYZ.Safe_Jog(3)

        Vgoal_L=0
        Vgoal_N=0
        Flag_Loop=0
        Samples_Rope_L = deque(maxlen=10)  # 自动丢弃旧数据
        Samples_Rope_L.append(0)
        Samples_Rope_S = deque(maxlen=3)  # 自动丢弃旧数据
        Samples_Rope_S.append(0)
        
        while True:
            dt=0.01
            time.sleep(dt)
            Rope_S=Srope.read_angles()
            Hand_S=Shand.read_angles()
            Samples_Rope_L.append(Rope_S)
            Ave_Rope_L=np.mean(Samples_Rope_L)
            Samples_Rope_S.append(Rope_S)
            Ave_Rope_S=np.mean(Samples_Rope_S)
            # print(Ave_Rope_L,Ave_Rope_S)

            if Hand_S>50:
                print('Measure Mode')
                mode=1
                Samples_BoxW=[]
                Vgoal=2000
                myXYZ.AxisMode_Jog(3,30,Vgoal)
                while Rope_S<400:
                    Rope_S=Srope.read_angles()
                    print('Waiting Lifting, Rope_S:',Rope_S)
                    time.sleep(0.02)
                i=0
                while i<40:
                    i=i+1
                    Rope_S=Srope.read_angles()
                    Samples_BoxW.append(Rope_S)
                    time.sleep(0.02)
                Weight=np.mean(Samples_BoxW)
                print(Weight)
                Flag_Loop=1
                Vgoal_L=Vgoal
            else:
                if Ave_Rope_L>500 and Flag_Loop==1:
                    mode=2 # Loop Mode
                    Rope_B=Weight
                    err=Ave_Rope_S-Rope_B
                    Vgoal_N=-5*err
                else:
                    mode=3 # Lossen Mode
                    Rope_B=30
                    err=Ave_Rope_S-Rope_B
                    Vgoal_N=-40*err
                diff=(Vgoal_N-Vgoal_L)*0.3
                Max_diff=300
                if diff>Max_diff:
                    diff=Max_diff
                if diff<-Max_diff:
                    diff=-Max_diff
                Vgoal=Vgoal_L+diff
                Max_Vel=4000
                Min_Vel=-2000
                if Vgoal<Min_Vel:
                    Vgoal=Min_Vel
                if Vgoal>Max_Vel:
                    Vgoal=Max_Vel
                myXYZ.AxisMode_Jog(3,30,Vgoal)
                Vgoal_L=Vgoal
                print('Mode:',mode,'Rope_S:', Rope_S,'Vgoal',Vgoal)

            # myXYZ.AxisMode_Jog(3,30,-1000)            

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        Shand.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


        

        