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
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        myXYZ.Safe_Jog(3)

        Vgoal_L=0
        Vgoal_N=0
        Samples_Rope_L = deque(maxlen=6)  # 自动丢弃旧数据
        Samples_Rope_S = deque(maxlen=3)  # 自动丢弃旧数据
        Samples_SW0_S=deque(maxlen=3)
        Samples_SW0_L=deque(maxlen=6)
        Flag_Weight=0
        Sig_Grasped=150

        while True:
            dt=0.01
            time.sleep(dt)
            Rope_S=Srope.read_angles()
            SW0=myXYZ.Read_IOs()

            Samples_Rope_L.append(Rope_S)
            Ave_Rope_L=np.mean(Samples_Rope_L)
            Samples_Rope_S.append(Rope_S)
            Ave_Rope_S=np.mean(Samples_Rope_S)
            Samples_SW0_S.append(SW0)
            Ave_SW0_S=np.mean(Samples_SW0_S)
            Samples_SW0_L.append(SW0)
            Ave_SW0_L=np.mean(Samples_SW0_L)
            
            if Ave_SW0_S>0.5 and Flag_Weight==0:
                print('Measure Mode')
                mode=1
                CurPos=myXYZ.Get_Pos(3)
                Vgoal=2000
                myXYZ.AxisMode_Jog(3,30,Vgoal)
                cur_time = time.time()
                travel_limit=800000
                while Rope_S<500 and (myXYZ.Get_Pos(3)-CurPos)<travel_limit and (time.time()-cur_time)<1.5:
                    Rope_S=Srope.read_angles()
                    print('Waiting Lifting, Rope_S:',Rope_S,'Travel:',(myXYZ.Get_Pos(3)-CurPos)/travel_limit,'Elapsed time:',time.time()-cur_time)
                    time.sleep(0.02)
                if (myXYZ.Get_Pos(3)-CurPos)<travel_limit and (time.time()-cur_time)<1.5:
                    Samples_BoxW=[]
                    i=0
                    while i<40:
                        i=i+1
                        Rope_S=Srope.read_angles()
                        Samples_BoxW.append(Rope_S)
                        time.sleep(0.02)
                    Weight=np.mean(Samples_BoxW)
                    print('Measured Success! Weight:',Weight)
                    Flag_Weight=1
                else:
                    print('Measured Failure: Over Travel or Time')
                    Vgoal=0
                    myXYZ.AxisMode_Jog(3,30,Vgoal)
                    time.sleep(1)
                Vgoal_L=Vgoal
            else:
                if Ave_SW0_L>0.3 and Flag_Weight==1:
                    mode=2 # Loop Mode
                    Rope_B=Weight
                    err=Ave_Rope_S-Rope_B
                    Vgoal_N=-15*err
                else:
                    mode=3 # Lossen Mode
                    Rope_B=30
                    err=Ave_Rope_S-Rope_B
                    Vgoal_N=-40*err
                    Flag_Weight=0
                
                diff=(Vgoal_N-Vgoal_L)*0.3
                Max_diff=180
                if diff>Max_diff:
                    diff=Max_diff
                if diff<-Max_diff:
                    diff=-Max_diff
                Vgoal=Vgoal_L+diff

                Max_Vel=4000
                Min_Vel=-2500
                if Vgoal<Min_Vel:
                    Vgoal=Min_Vel
                if Vgoal>Max_Vel:
                    Vgoal=Max_Vel
                myXYZ.AxisMode_Jog(3,30,Vgoal)
                Vgoal_L=Vgoal
                print('Mode:',mode, 'Rope_S:',Rope_S, 'Vgoal',int(Vgoal), 'Diff:',int(diff), 'SW0_S/L:',SW0,Ave_SW0_S,Ave_SW0_L)

            # myXYZ.AxisMode_Jog(3,30,-2500)            

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        Srope.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


        

        