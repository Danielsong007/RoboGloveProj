import sys
import time
from xyz_demo.xyz_utils import xyz_utils
from sensors.ruifen import RuiFenSensor
from sensors.dayang import DaYangSensor
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def main():
    try:
        Srope = DaYangSensor('/dev/ttyUSB0',0)
        ruifen = RuiFenSensor('/dev/ttyUSB2')
        myXYZ = xyz_utils()
        myXYZ.OpenEnableZero_ALL()
        InitPos=myXYZ.Safe_Jog()
        
        VgoalA1_L=0
        VgoalA2_L=0
        VgoalA3_L=0
        VgoalA3_N=0
        Samples_Rope_L = deque(maxlen=6)  # 自动丢弃旧数据
        Samples_Rope_S = deque(maxlen=3)  # 自动丢弃旧数据
        Samples_SW0_S=deque(maxlen=3)
        Samples_SW0_L=deque(maxlen=5)
        Flag_Weight=0

        while True:
            dt=0.01
            time.sleep(dt)
            SW0=myXYZ.Read_IOs()
            Rope_S=Srope.read_angles()
            x_angle, y_angle = ruifen.read_angles()

            Samples_Rope_L.append(Rope_S)
            Ave_Rope_L=np.mean(Samples_Rope_L)
            Samples_Rope_S.append(Rope_S)
            Ave_Rope_S=np.mean(Samples_Rope_S)
            Samples_SW0_S.append(SW0)
            Ave_SW0_S=np.mean(Samples_SW0_S)
            Samples_SW0_L.append(SW0)
            Ave_SW0_L=np.mean(Samples_SW0_L)

            if Ave_SW0_S>0.5 and Flag_Weight==0 and 1964000000-(myXYZ.Get_Pos(3)+InitPos)<4000000:
                print('Measure Mode')
                Mode_3=1
                CurPos=myXYZ.Get_Pos(3)
                VgoalA3=2000
                myXYZ.AxisMode_Jog(3,30,VgoalA3)
                cur_time = time.time()
                travel_limit=1000000
                while Rope_S<500 and (CurPos-myXYZ.Get_Pos(3))<travel_limit and (time.time()-cur_time)<1:
                    Rope_S=Srope.read_angles()
                    print('Waiting Lifting, Rope_S:',Rope_S,'Travel:',(CurPos-myXYZ.Get_Pos(3))/travel_limit,'Elapsed time:',time.time()-cur_time)
                    time.sleep(0.01)
                if (CurPos-myXYZ.Get_Pos(3))<travel_limit and (time.time()-cur_time)<1:
                    Samples_BoxW=[]
                    i=0
                    while i<50:
                        i=i+1
                        Rope_S=Srope.read_angles()
                        Samples_BoxW.append(Rope_S)
                        time.sleep(0.01)
                    Weight=np.mean(Samples_BoxW)
                    print('Measured Success! Weight:',Weight)
                    Flag_Weight=1
                else:
                    print('Measured Failure: Over Travel or Time')
                    VgoalA3=0
                    myXYZ.AxisMode_Jog(3,30,VgoalA3)
                    time.sleep(1.5)
                VgoalA3_L=VgoalA3
            else:
                if Ave_SW0_L>0.01 and Flag_Weight==1:
                    Mode_3=2 # Loop Mode
                    err=Weight-Ave_Rope_S
                    err_min=100
                    if err>-err_min and err<err_min:
                        err=0
                    VgoalA3_N=15*err
                else:
                    Mode_3=3 # Lossen Mode
                    err=40-Ave_Rope_S
                    VgoalA3_N=40*err
                    Flag_Weight=0
                
                diff=(VgoalA3_N-VgoalA3_L)*0.03
                Max_diff=150
                if diff>Max_diff:
                    diff=Max_diff
                if diff<-Max_diff:
                    diff=-Max_diff
                VgoalA3=VgoalA3_L+diff

                Max_Vel=4000
                Min_Vel=-4000
                if VgoalA3<Min_Vel:
                    VgoalA3=Min_Vel
                if VgoalA3>Max_Vel:
                    VgoalA3=Max_Vel
                myXYZ.AxisMode_Jog(3,30,VgoalA3)
                VgoalA3_L=VgoalA3
                # print('Mode_3:',Mode_3, 'Rope_S:',Rope_S, 'VgoalA3',int(VgoalA3), 'Err',round(err,2), 'Diff:',int(diff), 'SW0_S/L:',SW0,round(Ave_SW0_S,3),round(Ave_SW0_L,3))
            
            if abs(y_angle) < 2.5:
                VgoalA1=-30*y_angle
                Mode_1='Slow'
            elif abs(y_angle) < 10:
                VgoalA1=-120*y_angle
                Mode_1='Fast'
            else:
                VgoalA1=VgoalA1_L
                Mode_1='Last'

            if abs(x_angle) < 2.5:
                VgoalA2=-20*x_angle
                Mode_2='Slow'
            elif abs(x_angle) < 10:
                VgoalA2=-90*x_angle
                Mode_2='Fast'
            else:
                VgoalA2=VgoalA2_L
                Mode_2='Last'
            print('x_angle:',round(x_angle,2), 'y_angle:',round(y_angle,2), 'Mode_1',Mode_1, 'Mode_2',Mode_2, 'VgoalA1',round(VgoalA1,2), 'VgoalA2',round(VgoalA2,2))
            VgoalA1Max=700
            VgoalA2Max=350
            if VgoalA1>VgoalA1Max:
                VgoalA1=VgoalA1Max
            if VgoalA1<-VgoalA1Max:
                VgoalA1=-VgoalA1Max
            if VgoalA2>VgoalA2Max:
                VgoalA2=VgoalA2Max
            if VgoalA2<-VgoalA2Max:
                VgoalA2=-VgoalA2Max
            myXYZ.AxisMode_Jog(1,4,VgoalA1)
            myXYZ.AxisMode_Jog(2,1,VgoalA2)
            VgoalA1_L=VgoalA1
            VgoalA2_L=VgoalA2

    except KeyboardInterrupt:
        print("Ctrl-C is pressed!")
        
    finally:
        myXYZ.SafeQuit()
        ruifen.close()
        Srope.close()
        sys.exit(0)

if __name__ == "__main__":
    main()


        

        